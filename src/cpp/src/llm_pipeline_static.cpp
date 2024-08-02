// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm_pipeline_static.hpp"

#include "openvino/opsets/opset13.hpp"

#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include <openvino/pass/serialize.hpp>
#include <openvino/openvino.hpp>
#include <filesystem>
#include <fstream>
#include <variant>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <random>

// #include <openvino/pass/stateful_to_stateless.hpp>

namespace {

std::shared_ptr<ov::Model> add_slices_to_kvcache_inputs(const std::shared_ptr<ov::Model>& model) {
    const auto kvcache_name_pattern = "past_key_values";
    std::vector<std::shared_ptr<ov::opset13::Parameter>> new_params;
    for (auto param : model->get_parameters()) {
        auto tensor_name = param->get_output_tensor(0).get_any_name();
        if (tensor_name.find(kvcache_name_pattern) == std::string::npos) {
            new_params.push_back(param);
            continue;
        }
        auto shape = param->get_output_shape(0);
        shape[2] += 1;

        auto new_param = std::make_shared<ov::opset13::Parameter>(param->get_element_type(), shape);
        new_param->set_friendly_name(tensor_name);
        new_param->outputs().begin()->get_tensor().set_names(param->outputs().begin()->get_tensor().get_names());

        auto slice_start = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{1}
        );
        auto slice_stop = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{static_cast<int32_t>(shape[2])}
        );
        auto slice_step = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{1}
        );
        auto slice_axes = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{2}
        );
        auto slice_node = std::make_shared<ov::opset13::Slice>(
            new_param, slice_start->output(0), slice_stop->output(0), slice_step->output(0), slice_axes->output(0)
        );
        slice_node->set_friendly_name(tensor_name + "_Slice");
        for (auto target_input : param->output(0).get_target_inputs()) {
            target_input.replace_source_output(slice_node->output(0));
        }
        new_params.push_back(new_param);
    }
    return std::make_shared<ov::Model>(model->get_results(), ov::SinkVector{}, new_params);
}

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = ov::PartialShape({1,
                                          partial_shape[1].get_length(),
                                          kvcache_size-input_size,
                                          partial_shape[3].get_length()});
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

static inline void fill_zero(ov::Tensor& tensor) {
    size_t tensor_size = tensor.get_size();
    if (0 == tensor_size) {
        throw std::runtime_error(
            "Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    }
    ov::float16* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = 0;
    }
}

void fill_tensor(ov::Tensor tensor, int64_t fill_val, size_t offset = 0u) {
    ov::float16* tensor_data = tensor.data<ov::float16>();
    std::fill(tensor_data + offset, tensor_data + tensor.get_size(), (ov::float16)fill_val);
}

void copy_with_offset(const ov::Tensor& orig, const int32_t offset, ov::Tensor& padded) {
    int64_t* orig_data = orig.data<int64_t>();
    ov::float16* padded_data = padded.data<ov::float16>();
    std::copy(orig_data, orig_data + orig.get_size(), padded_data + offset);
}

ov::AnyMap extract_config_or_default(const ov::AnyMap& config, const std::string& config_name) {
    ov::AnyMap stage_cfg;
    if (auto it = config.find(config_name); it != config.end()) {
        const auto& map = it->second.as<std::map<std::string, std::string>>();
        stage_cfg = { map.begin(), map.end() };
    } else if (config_name == "PREFILL_CONFIG") {
        std::map<std::string, std::string> prefill_config = {
            { "NPU_USE_NPUW", "YES" },
            { "NPUW_FOLD", "YES" },
            { "NPUW_DCOFF_TYPE", "f16" },
            { "NPUW_DCOFF_SCALE",  "YES" },
            { "NPUW_ONLINE_AVOID", "P:RMSNorm/NPU" }
        };
        stage_cfg.insert(prefill_config.begin(), prefill_config.end());
    } else if (config_name == "GENERATE_CONFIG") {
        std::map<std::string, std::string> generate_config = {
            { "NPU_USE_NPUW", "YES" },
            { "NPUW_FOLD", "YES" },
            { "NPUW_DCOFF_TYPE", "f16" },
            { "NPUW_DCOFF_SCALE", "YES" },
            { "NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add" },
            { "NPUW_PARALLEL_COMPILE", "YES" },
            { "NPUW_FUNCALL_ASYNC", "YES" }
        };
        stage_cfg.insert(generate_config.begin(), generate_config.end());
    }
    return stage_cfg;
}

} // anonymous namespace

namespace ov {
namespace genai {

StaticLLMPipeline::StaticLLMPipeline(
    const std::filesystem::path& path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& config
) : LLMPipelineImplBase(tokenizer,
                        utils::from_config_json_if_exists(path)) {
    ov::Core core;
    // (1) Read the template model - this will be kvcache model
    std::ifstream modelStream("C:\\WorkSpace\\Yihan\\models\\Phi-3\\ovmergev6\\vpuip.blob", std::ios_base::binary | std::ios_base::in);
    if (!modelStream.is_open()) {
        std::cout << "Cannot open kvcache blob file" << std::endl;
    }
    std::cout << "Use kvcache blob file" << std::endl;
    ov::AnyMap latency{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};
    m_kvcache_request = core.import_model(modelStream, device, latency).create_infer_request();
    std::cout << "kvcache blob file imported on " << device << std::endl;
    modelStream.close();

    // (4) Reshape both models to static shape
    m_kvcache_desc = KVCacheDesc { 1024u, 0u };
    const uint32_t max_prompt_size = m_kvcache_desc.total_size;
    const uint32_t max_kvcache_size = m_kvcache_desc.total_size;
};

StaticLLMPipeline::StaticLLMPipeline(
    const std::filesystem::path& path,
    const std::string& device,
    const ov::AnyMap& config
) : StaticLLMPipeline(path, path.string(), device, config) {
}

void StaticLLMPipeline::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void StaticLLMPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

void StaticLLMPipeline::prepare_for_new_conversation() {
    fill_tensor(m_prefill_request.get_tensor("input_ids"), m_tokenizer.get_pad_token_id());
    fill_tensor(m_prefill_request.get_tensor("position_ids"), 0u);
    fill_tensor(m_prefill_request.get_tensor("attention_mask"), 0u);
    fill_tensor(m_kvcache_request.get_tensor("attention_mask"), 0u);
    m_kvcache_desc.num_stored_tokens = 0u;
}

DecodedResults StaticLLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {

    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    if (std::holds_alternative<std::vector<std::string>>(inputs)) {
        OPENVINO_THROW("Currently only batch size=1 is supported");
    }

    OPENVINO_ASSERT(std::holds_alternative<std::string>(inputs));
    auto& prompt = std::get<std::string>(inputs);

    if (m_is_chat_conversation) {
        m_history.push_back({{"role", "user"}, {"content", prompt}});
        constexpr bool add_generation_prompt = true;
        prompt = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
    }

    auto tokenized_input = m_tokenizer.encode(prompt);
    auto encoded_results = generate(tokenized_input, config, streamer);
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};

    if (m_is_chat_conversation) {
        auto answer = decoded_results.texts[0];
        m_history.push_back({{"role", "assistant"}, {"content", answer}});
    }
    return decoded_results;
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

inline double get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
};

inline double get_duration_ms(Time::time_point& startTime, Time::time_point& endTime) {
    return std::chrono::duration_cast<ns>(endTime - startTime).count() * 0.000001;
};

void copy_with_left_offset(const ov::Tensor& orig, ov::Tensor& padded) {
    const auto orig_size = orig.get_size();
    const auto padded_size = padded.get_size();
    const auto kLeftOffset = padded_size - orig_size;
    int64_t* orig_data = orig.data<int64_t>();
    ov::float16* padded_data = padded.data<ov::float16>();
    std::copy(orig_data, orig_data + orig_size, padded_data + kLeftOffset);
}
bool ends_with(const std::string& str, const std::string& suffix) {
    // Check if the length of the suffix is greater than the string itself
    if (suffix.size() > str.size()) return false;

    // Use rfind to find the suffix at the end of the string
    return str.rfind(suffix) == (str.size() - suffix.size());
}

EncodedResults StaticLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    const auto& compiledModel = m_kvcache_request.get_compiled_model();

    ov::element::Type layer_type = ov::element::f16;
    
    for (const ov::Output<const ov::Node>& model_input : compiledModel.inputs()) {
        fill_zero(m_kvcache_request.get_tensor(model_input));
    }

    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    
    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = data->input_ids;
        attention_mask = data->attention_mask;
    }

    if (input_ids.get_shape().at(0) > 1u) {
        OPENVINO_THROW("Currently only batch size=1 is supported");
    }

    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    std::shared_ptr<StreamerBase> streamer_ptr;
    if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
        streamer_ptr = nullptr;
    } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
        streamer_ptr = *streamer_obj;
    } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
        streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
    }

    if (!config.is_greedy_decoding()) {
        OPENVINO_THROW("Currently only greedy decoding is supported");
    }

    ov::genai::EncodedResults results;
    // NB: Only batch=1 is supported now
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);

    // NB: Check if input prompt less than maximum size
    auto prompt_len = input_ids.get_size();
    if (prompt_len > m_kvcache_desc.total_size) {
        OPENVINO_THROW("Currently static pipeline only process up to " + std::to_string(m_kvcache_desc.total_size) + " tokens");
    }

    // dimension for phi-3 model
    size_t N = 1;
    size_t H = 32;
    size_t S = 1023;
    size_t D = 96;

    auto start_time = Time::now();

    // auto padded_attention_mask = m_kvcache_request.get_tensor("attention_mask");
    // copy_with_left_offset(attention_mask, padded_attention_mask);

    ov::float16 new_token;
    // NB: Prefill stage
    ov::float16* input_ids_data = m_kvcache_request.get_tensor("input_ids").data<ov::float16>();
    ov::float16* position_ids_data = m_kvcache_request.get_tensor("position_ids").data<ov::float16>();
    ov::float16* attention_mask_data = m_kvcache_request.get_tensor("attention_mask").data<ov::float16>();

    const auto kStartInputKVCacheLayers = 0u;
    const auto kStartOutputKVCacheLayers = 1u;

    int64_t* orig_data = input_ids.data<int64_t>();
    for (int i = 0; i < prompt_len; ++i) {
        new_token = orig_data[m_kvcache_desc.total_size - prompt_len + i];
        input_ids_data[0] = new_token;
        position_ids_data[0] = i;
        size_t attention_tmp_idx = i - 1;
        if (i == 0) {
            attention_tmp_idx = m_kvcache_desc.total_size - i - 1;
        }
        attention_mask_data[attention_tmp_idx] = 1u;

        m_kvcache_request.infer();

        for (int j = 0; j < compiledModel.outputs().size() - 1; ++j) {
            // std::string input_name = compiledModel.inputs()[kStartInputKVCacheLayers + j].get_any_name();
            std::string output_name = compiledModel.outputs()[kStartOutputKVCacheLayers + j].get_any_name();
            auto kvcache_out_tensor = m_kvcache_request.get_tensor(output_name);
            ov::float16 *outputs_data = kvcache_out_tensor.data<ov::float16>();
            
            if (ends_with(output_name, "value")) {
                std::string input_name = output_name.replace(output_name.find("present"), 7, "past_key_values");
                auto kvcache_tensor = m_kvcache_request.get_tensor(input_name);
                ov::float16 *inputs_data = kvcache_tensor.data<ov::float16>();
                // NHDS
                for (size_t n = 0; n < N; ++n) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t d= 0; d < D; ++d) {
                            size_t data_index = n * (H * D * 1) + h * (D * 1) + d * 1;
                            size_t input_index = n * (H * D * S) + h * (D * S) + d * S + i;
                            inputs_data[input_index] = outputs_data[data_index];
                        }
                    }
                }
            } else {
                std::string input_name = output_name.replace(output_name.find("present"), 7, "past_key_values");
                auto kvcache_tensor = m_kvcache_request.get_tensor(input_name);
                ov::float16 *inputs_data = kvcache_tensor.data<ov::float16>();
                // NHSD
                for (size_t n = 0; n < N; ++n) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t s= 0; s < S; ++s) {
                            size_t data_index = n * (H * S * 1) + h * (S * 1) + s * 1;
                            size_t input_index = n * (H * S * D) + h * (S * D) + s * D + i;
                            inputs_data[input_index] = outputs_data[data_index];
                        }
                    }
                }
            }
        }
    }

    // Generation stage
    m_kvcache_desc.num_stored_tokens += prompt_len;
    ov::float16 last_token = utils::argmax(m_kvcache_request.get_tensor("logits"), 0);
    results.tokens[0].push_back(last_token);
    if (streamer_ptr && streamer_ptr->put(last_token)) {
        return results;
    }

    // padded_attention_mask.copy_to(m_kvcache_request.get_tensor("attention_mask"));
    auto first_token_time = Time::now();
    const size_t max_tokens = config.get_max_new_tokens(prompt_len);
    // for (int i = 0; i < max_tokens - 1; ++i) {
    for (int i = prompt_len; i < max_tokens; ++i) {
        input_ids_data[0] = (ov::float16)last_token;
        position_ids_data[0] = (ov::float16)i;
        attention_mask_data[i -1] = (ov::float16)1u;

        m_kvcache_request.infer();

         for (int j = 0; j < compiledModel.outputs().size() - 1; ++j) {
            // std::string input_name = compiledModel.inputs()[kStartInputKVCacheLayers + j].get_any_name();
            std::string output_name = compiledModel.outputs()[kStartOutputKVCacheLayers + j].get_any_name();
            auto kvcache_out_tensor = m_kvcache_request.get_tensor(output_name);
            ov::float16 *outputs_data = kvcache_out_tensor.data<ov::float16>();
            
            if (ends_with(output_name, "value")) {
                std::string input_name = output_name.replace(output_name.find("present"), 7, "past_key_values");
                auto kvcache_tensor = m_kvcache_request.get_tensor(input_name);
                ov::float16 *inputs_data = kvcache_tensor.data<ov::float16>();
                // NHDS
                for (size_t n = 0; n < N; ++n) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t d= 0; d < D; ++d) {
                            size_t data_index = n * (H * D * 1) + h * (D * 1) + d * 1;
                            size_t input_index = n * (H * D * S) + h * (D * S) + d * S + i;
                            inputs_data[input_index] = outputs_data[data_index];
                        }
                    }
                }
            } else {
                std::string input_name = output_name.replace(output_name.find("present"), 7, "past_key_values");
                auto kvcache_tensor = m_kvcache_request.get_tensor(input_name);
                ov::float16 *inputs_data = kvcache_tensor.data<ov::float16>();
                // NHSD
                for (size_t n = 0; n < N; ++n) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t s= 0; s < S; ++s) {
                            size_t data_index = n * (H * S * 1) + h * (S * 1) + s * 1;
                            size_t input_index = n * (H * S * D) + h * (S * D) + s * D + i;
                            inputs_data[input_index] = outputs_data[data_index];
                        }
                    }
                }
            }
        }

        last_token = utils::argmax(m_kvcache_request.get_tensor("logits"), 0);
        // last_token = 0;
        results.tokens[0].push_back((int64_t)last_token);

        if (streamer_ptr && streamer_ptr->put(last_token)) {
            break;
        }

        if ((int64_t)last_token == m_generation_config.eos_token_id) {
            break;
        }

        // NB: KV-cache is full, further generation is impossible
        if (m_kvcache_desc.num_stored_tokens == m_kvcache_desc.total_size) {
            break;
        }
    }

    int token_count = results.tokens[0].size();
    
    auto end_time = Time::now();
    std::cout << "Performance metrics: " << std::endl;
    std::cout << "Generated tokens: " << token_count << std::endl;
    std::cout << "First token generation time: " << get_duration_ms(start_time, first_token_time) << std::endl;
    std::cout << "Generation average latency: " << get_duration_ms(first_token_time, end_time)/(token_count - 1) << std::endl;
    std::cout << "token/s: " << (token_count - 1)/(get_duration_ms(first_token_time, end_time)/1000) << std::endl;
    std::cout << "Generation time: " << get_duration_ms(start_time, end_time) << std::endl;

    std::cout << "Tokens:\n";
    for (const auto& token_vector : results.tokens) {
        std::cout << "[";
        for (size_t i = 0; i < token_vector.size(); ++i) {
            std::cout << token_vector[i];
            if (i != token_vector.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }

    std::cout << "Scores:\n[";
    for (size_t i = 0; i < results.scores.size(); ++i) {
        std::cout << results.scores[i];
        if (i != results.scores.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
    return results;
}

}  // namespace genai
}  // namespace ov
