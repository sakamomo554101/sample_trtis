#include "src/custom/sdk/custom_instance.h"
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>

#define LOG_INFO std::cout

// header
namespace nvidia {
    namespace inferenceserver {
        namespace custom {
            namespace sample_instance {
                enum ErrorCodes {
                    kSuccess = 0,
                    kBatching,
                    kInputContents,
                    kError
                };

                class Context : public CustomInstance {
                    public:
                        Context(
                            const std::string& instance_name, const ModelConfig& config,
                            const int gpu_device, const size_t server_parameter_cnt,
                            const char** server_parameters);

                        int Init();

                        int Execute(
                            const uint32_t payload_cnt, CustomPayload* payloads,
                            CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

                    private:
                        const int kBatching = RegisterError("Batching is not supported!");
                        const int kInputContents = RegisterError("input error!");
                        const int kError = RegisterError("unknown Error!");
                };

                Context::Context(
                    const std::string& instance_name, const ModelConfig& model_config,
                    const int gpu_device, const size_t server_parameter_cnt,
                    const char** server_parameters)
                    : CustomInstance(instance_name, model_config, gpu_device)
                {
                    LOG_INFO << "[SampleInstance] call constructor" << std::endl;
                }

                int
                Context::Init()
                {
                    LOG_INFO << "[SampleInstance] start to init" << std::endl;
                    return kSuccess;
                }

                int
                Context::Execute(
                    const uint32_t payload_cnt, CustomPayload* payloads,
                    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
                {
                    LOG_INFO << "[SampleInstance] start to execute!" << std::endl;
                    size_t output_cnt = 0;
                    std::string output;

                    // get input data
                    {
                        // get input name
                        const char* input_name = payloads[0].input_names[0];

                        // get input data from request
                        const void* content;
                        uint64_t content_byte_size = -1;
                        if (!input_fn(payloads[0].input_context, input_name, &content, &content_byte_size)) {
                            LOG_INFO << "[SampleInstance] error input_fn" << std::endl;
                            return kInputContents;
                        }

                        // If 'content' returns nullptr or if the content is not the
                        // expected size, then something went wrong.
                        if (content == nullptr) {
                            LOG_INFO << "[SampleInstance] error input content is null" << std::endl;
                            return kInputContents;
                        }

                        // get input size
                        uint32_t input_byte_size = 0;
                        uint64_t byte_to_append = 4; // TODO : get byte to append from input content
                        memcpy(&input_byte_size, static_cast<const char*>(content), byte_to_append);

                        // extract input text
                        auto value = std::string(
                            (uint32_t*)content, 
                            (uint32_t*)content + (content_byte_size/sizeof(uint32_t))
                        );
                        uint32_t byte_size = value.size();
                        output.append(reinterpret_cast<const char*>(&byte_size), 4);
                        output.append(value);
                        output_cnt++;
                    }

                    // get output name
                    const char* output_name = payloads[0].required_output_names[0];

                    // get output shape
                    std::vector<int64_t> output_shape;
                    output_shape.push_back(output_cnt);
                    output_shape.insert(output_shape.begin(), 1);  // TODO : get batch size from payload

                    // allocate memory of output
                    void* obuffer;
                    if (!output_fn(
                        payloads[0].output_context, output_name, output_shape.size(),
                        &output_shape[0], output.size(), &obuffer)) {
                        LOG_INFO << "[SampleInstance] error to allocate memory" << std::endl;
                            return kError;
                    }

                    // copy data into output buffer
                    if (obuffer != nullptr) {
                        memcpy(obuffer, output.c_str(), output.size());
                    }
                    return kSuccess;
                }
            }

            // Creates a new sample instance context instance
            int
            CustomInstance::Create(
                CustomInstance** instance, const std::string& name,
                const ModelConfig& model_config, int gpu_device,
                const CustomInitializeData* data)
            {
                sample_instance::Context* context = new sample_instance::Context(
                    name, model_config, gpu_device, data->server_parameter_cnt,
                    data->server_parameters);

                *instance = context;

                if (context == nullptr) {
                    LOG_INFO << "[SampleInstance] context is nullptr" << std::endl;
                    return ErrorCodes::CreationFailure;
                }
                return context->Init();
            }
        }
    }
}