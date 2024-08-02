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

template <typename T>
using uniformDistribution = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

template <typename T, typename T2>
static inline void fill_random(ov::Tensor& tensor,
                               T rand_min = std::numeric_limits<uint8_t>::min(),
                               T rand_max = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(0);
    size_t tensor_size = tensor.get_size();
    if (0 == tensor_size) {
        throw std::runtime_error(
            "Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    }
    ov::float16* data = tensor.data<ov::float16>();
    uniformDistribution<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = static_cast<ov::float16>(distribution(gen));
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
    // fill_tensor(m_prefill_request.get_tensor("input_ids"), m_tokenizer.get_pad_token_id());
    // fill_tensor(m_prefill_request.get_tensor("position_ids"), 0u);
    // fill_tensor(m_prefill_request.get_tensor("attention_mask"), 0u);
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

static inline std::string double_to_string(const double number) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << number;
    return ss.str();
}

EncodedResults StaticLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    const auto& compiledModel = m_kvcache_request.get_compiled_model();
    ov::element::Type layer_type = ov::element::f16;
    
    // for (int j = 0; j < compiledModel.inputs().size() - 1; ++j) {
    //     std::string input_name = compiledModel.inputs()[j].get_any_name();
    //     ov::Tensor zero_tensor = ov::Tensor(layer_type, ov::Shape(compiledModel.inputs()[j].get_shape()));
    //     m_kvcache_request.set_tensor(input_name, zero_tensor);
    // }
    for (const ov::Output<const ov::Node>& model_input : compiledModel.inputs()) {
        fill_random<short, short>(m_kvcache_request.get_tensor(model_input));
    }

    auto startTime = Time::now();
 
    m_kvcache_request.infer();
    auto duration_ms = get_duration_ms_till_now(startTime);
    std::cout << "First inference took " << double_to_string(duration_ms) << " ms" << std::endl;
    
    size_t iteration = 0;
    startTime = Time::now();
    while ( iteration < 100) {
        // for (int j = 0; j < compiledModel.inputs().size() - 1; ++j) {
        //     std::string input_name = compiledModel.inputs()[j].get_any_name();
        //     ov::Tensor zero_tensor = ov::Tensor(layer_type, ov::Shape(compiledModel.inputs()[j].get_shape()));
        //     m_kvcache_request.set_tensor(input_name, zero_tensor);
        // }
        m_kvcache_request.infer();
        ++iteration;
    }
    auto totalDuration = get_duration_ms_till_now(startTime);
    double fps = 1000.0 * iteration / totalDuration;
    std::cout << "Count:               " << iteration << " iterations" << std::endl;
    std::cout << "Duration:            " << double_to_string(totalDuration) << " ms" << std::endl;
    std::cout << "Throughput:          " << double_to_string(fps) << " FPS" << std::endl;

    ov::genai::EncodedResults results;
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);
    return results;
}

}  // namespace genai
}  // namespace ov
