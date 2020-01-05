#include <chrono>
#include <string>
#include <thread>
#include <map>

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

namespace nvidia { namespace inferenceserver { namespace custom {
namespace sample_sequence {

// Context object. All state must be kept in this object.
class Context : public CustomInstance {
 public:
  Context(
      const std::string& instance_name, const ModelConfig& config,
      const int gpu_device);
  ~Context();

  int Init();

  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  int GetInputTensor(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, std::vector<uint8_t>* input);
  
  int SaveInputText(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name, uint64_t corrid);
  
  int OutputText(
      CustomGetOutputFn_t output_fn, CustomPayload* payloads, std::string output_text); 
  
  void PrintText(
      std::map<uint64_t, std::vector<std::string>> texts);

  // save text data from each client
  std::map<uint64_t, std::vector<std::string>> saved_texts;

  // Local error codes
  const int kGpuNotSupported = RegisterError("execution on GPU not supported");
  const int kSequenceBatcher =
      RegisterError("model configuration must configure sequence batcher");
  const int kModelControl = RegisterError(
      "'START' and 'READY' must be configured as the control inputs");
  const int kInputOutput =
      RegisterError("model must have two inputs and one output with shape [1]");
  const int kInputName = RegisterError("model input must be named 'INPUT'");
  const int kOutputName = RegisterError("model output must be named 'OUTPUT'");
  const int kInputOutputDataType =
      RegisterError("model input and output must have TYPE_INT32 data-type");
  const int kInputContents = RegisterError("unable to get input tensor values");
  const int kInputSize = RegisterError("unexpected size for input tensor");
  const int kOutputBuffer =
      RegisterError("unable to get buffer for output tensor values");
  const int kBatchTooBig =
      RegisterError("unable to execute batch larger than max-batch-size");
  const int kTimesteps =
      RegisterError("unable to execute more than one timestep at a time");
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const int gpu_device)
    : CustomInstance(instance_name, model_config, gpu_device)
{
}

Context::~Context() {}

int
Context::Init()
{
  // gpu is not supported
  if (gpu_device_ != CUSTOM_NO_GPU_DEVICE) {
    return kGpuNotSupported;
  }

  // only sequence batcher
  if (!model_config_.has_sequence_batching()) {
    return kSequenceBatcher;
  }

  // check the sequence batching config
  auto& batcher = model_config_.sequence_batching();
  if (batcher.control_input_size() != 4) {
    // set control input count is 4(START, READY, END, CORRID)
    return kModelControl;
  }
  // TODO : check the control input names

  // TODO : check the other config
  return ErrorCodes::Success;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  std::cout << "[SampleSequence] start to execute....";
  int err;

  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];
    if (payload.batch_size != 1) {
      payload.error_code = kTimesteps;
      continue;
    }

    // get the control inputs(START, END, READY, CORRID)
    // byte sizeでデータを取得するため、uint8のバッファに格納してから、各データの型に合わせて、変換する
    size_t batch1_control_input_size = GetDataTypeByteSize(TYPE_INT32);
    std::vector<uint8_t> start_buffer, end_buffer, ready_buffer, corrid_buffer;
    err = GetInputTensor(
        input_fn, payload.input_context, "START", batch1_control_input_size,
        &start_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    err = GetInputTensor(
        input_fn, payload.input_context, "END", batch1_control_input_size,
        &end_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    err = GetInputTensor(
        input_fn, payload.input_context, "READY", batch1_control_input_size,
        &ready_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    batch1_control_input_size = GetDataTypeByteSize(TYPE_UINT64);
    err = GetInputTensor(
        input_fn, payload.input_context, "CORRID", batch1_control_input_size,
        &corrid_buffer);
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    int32_t* start = reinterpret_cast<int32_t*>(&start_buffer[0]);
    int32_t* end = reinterpret_cast<int32_t*>(&end_buffer[0]);
    int32_t* ready = reinterpret_cast<int32_t*>(&ready_buffer[0]);
    uint64_t* corrid = reinterpret_cast<uint64_t*>(&corrid_buffer[0]);

    std::cout << "[SampleSequence] start is " << *start 
              << " : end is " << *end 
              << " : ready is " << *ready 
              << " : corrid is " << *corrid << std::endl;
    
    // get the input data of text if needed
    if (*start == 1) {
      // add start text(this text is not returned)
      saved_texts[*corrid].push_back("start");
    } else if (*end == 1) {
      // concat texts of saved_texts vector
      std::string output_text;
      saved_texts[*corrid].erase(saved_texts[*corrid].begin());
      for (auto text : saved_texts[*corrid]) {
        output_text.append(text);
      }

      // reset the text datas of corrid
      saved_texts[*corrid].clear();

      // return the output text
      return OutputText(output_fn, payloads, output_text);
    } else {
      // if "start" text is already added, text will be added.
      if (saved_texts[*corrid].size() > 0 && saved_texts[*corrid].front() == "start") {
        const char* input_name = payload.input_names[0];
        err = SaveInputText(
            input_fn, payload.input_context, input_name, *corrid
        );
        if (err != ErrorCodes::Success) {
          payload.error_code = err;
          continue;
        }
      }
    }
  }

  // if end token is not received, blank text is returned.
  return OutputText(output_fn, payloads, "");
}

int
Context::OutputText(
    CustomGetOutputFn_t output_fn, CustomPayload* payloads, std::string output_text) 
{
  // set output
  std::string output;
  uint32_t byte_size = output_text.size();
  output.append(reinterpret_cast<const char*>(&byte_size), 4);
  output.append(output_text);

  // get output name
  const char* output_name = payloads[0].required_output_names[0];

  // get output shape
  std::vector<int64_t> output_shape;
  output_shape.push_back(1);
  output_shape.insert(output_shape.begin(), 1);

  // allocate memory of output
  void* obuffer;
  if (!output_fn(
      payloads[0].output_context, output_name, output_shape.size(),
      &output_shape[0], output.size(), &obuffer)) {
        LOG_INFO << "[SampleSequence] error to allocate memory" << std::endl;
        return kOutputBuffer;
  }

  // copy data into output buffer
  if (obuffer != nullptr) {
      memcpy(obuffer, output.c_str(), output.size());
  }
  return ErrorCodes::Success;
}

int
Context::SaveInputText(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* input_name, uint64_t corrid)
{
  const void* content;
  uint64_t content_byte_size = -1;
  if (!input_fn(input_context, input_name, &content, &content_byte_size)) {
      LOG_INFO << "[SampleSequence] error input_fn" << std::endl;
      return kInputContents;
  }

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
  saved_texts[corrid].push_back(value);

  return ErrorCodes::Success;
}

int
Context::GetInputTensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, std::vector<uint8_t>* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input' vector. A
  // more performant solution would attempt to use the input tensors
  // in-place instead of having this copy.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size = expected_byte_size - total_content_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    std::cout << std::string(name) << ": size " << content_byte_size << ", "
              << (reinterpret_cast<const int32_t*>(content)[0]) << std::endl;

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    total_content_byte_size += content_byte_size;
    if (total_content_byte_size > expected_byte_size) {
      return kInputSize;
    }

    input->insert(
        input->end(), static_cast<const uint8_t*>(content),
        static_cast<const uint8_t*>(content) + content_byte_size);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return ErrorCodes::Success;
}

void
Context::PrintText(std::map<uint64_t, std::vector<std::string>> texts) {
  for (auto texts_with_key : texts) {
    std::cout << "[SampleSequence] corrid is " << texts_with_key.first << std::endl;
    for (std::string text : texts_with_key.second) {
      std::cout << "[SampleSequence] text is " + text << std::endl;
    }
  }
}

}  // namespace sample_sequence

// Creates a new sample_sequence context instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
  sample_sequence::Context* context =
      new sample_sequence::Context(name, model_config, gpu_device);

  *instance = context;

  if (context == nullptr) {
    return ErrorCodes::CreationFailure;
  }

  return context->Init();
}

}}}  // namespace nvidia::inferenceserver::custom
