#include <chrono>
#include <string>
#include <thread>
#include <map>

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"
#include "custom_backend/face_recognition_model/face_wrapper.h"

// dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// boost
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>

using namespace boost::property_tree;

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

namespace nvidia { namespace inferenceserver { namespace custom {
namespace face_recognition_model {

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
  const std::string INSTANCE_NAME = "[face_recognition_model]";
  std::map<uint64_t, std::vector<std::vector<face_recognition_model::FaceData>>> data_map;
  face_recognition_model::FaceWrapper face_wrapper;

  int GetControlInput(
      CustomGetNextInputFn_t input_fn, void* input_context, 
      int32_t& start, int32_t& end, int32_t& ready, uint64_t& corrid);

  int GetInputTensor(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, std::vector<uint8_t>* input);  
  
  int GetInputMatrix(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* input_name, uint64_t corrid, cv::Mat& input_matrix);
  
  int Output(
    CustomGetOutputFn_t output_fn, CustomPayload* payloads, std::string output_text);
  
  std::string CreateResultTreeStr(std::vector<std::vector<face_recognition_model::FaceData>> face_detect_datas);

  // local error codes
  const int kInputContents = RegisterError("unable to get input tensor values");
  const int kInputSize = RegisterError("unexpected size for input tensor"); 
  const int kOutputBuffer = RegisterError("unable to get buffer for output tensor values");

  // debug mode
  const bool kDebugMode = true;
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
  // 顔認証ライブラリのインスタンスをキャッシュする
  //face_wrapper = face_recognition_model::FaceWrapper();
  face_wrapper.Initialize();
  return ErrorCodes::Success;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  std::cout << INSTANCE_NAME << " start to execute.." << std::endl;
  int err;

  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    // get control input
    int32_t start, end, ready;
    uint64_t corrid;
    err = GetControlInput(
      input_fn, payload.input_context, start, end, ready, corrid
    );
    if (err != ErrorCodes::Success) {
      payload.error_code = err;
      continue;
    }

    // if start token is received, check the data_map
    if (start == 1) {
      if (data_map.count(corrid) == 0) {
        // not received "start" token
        data_map[corrid] = {};
      }
      continue;
    }
    if (end == 1) {
      if (data_map.count(corrid) == 0) {
        // nothing to do
        continue;
      }

      // push result into result list
      auto result = CreateResultTreeStr(data_map[corrid]);

      // delete data
      data_map.erase(corrid);

      // return json result
      return Output(output_fn, payloads, result);
    }
    // if start token was already received and image data currently is received, do face recognition process
    if (data_map.count(corrid) > 0) {
      // start to do face recognition process
      cv::Mat cv_img;
      const char* input_name = payload.input_names[0];
      err = GetInputMatrix(input_fn, payload.input_context, input_name, corrid, cv_img);

      // detect and recognize face data
      auto face_datas = face_wrapper.ExtractFaceData(cv_img);
      if (face_datas.empty()) {
        std::cout << INSTANCE_NAME << " face data is not found" << std::endl;
        continue;
      }
      data_map[corrid].push_back(face_datas);
    }
  }

  std::cout << INSTANCE_NAME << " end to execute.." << std::endl;
  return Output(output_fn, payloads, "");
}

int
Context::GetInputMatrix(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* input_name, uint64_t corrid, cv::Mat& input_matrix)
{
  const void* content;
  uint64_t content_byte_size = -1;
  if (!input_fn(input_context, input_name, &content, &content_byte_size)) {
      LOG_INFO << INSTANCE_NAME << " error input_fn" << std::endl;
      return kInputContents;
  }

  if (content == nullptr) {
      LOG_INFO << INSTANCE_NAME << " error input content is null" << std::endl;
      return kInputContents;
  }
  std::cout << INSTANCE_NAME << " byte size is " << content_byte_size << std::endl;

  // set Matrix from content ( TODO : get witdh and height from Model config)
  cv::Mat tmp_matrix(480, 640, CV_8UC3, (void*)content);
  input_matrix = tmp_matrix;
  std::cout << INSTANCE_NAME << " input_matrix : dims " << input_matrix.dims 
                             << ", depth(byte/channel) " << input_matrix.elemSize1() 
                             << ", channels: " << input_matrix.channels()
                             << ", rows: " << input_matrix.rows
                             << ", cols: " << input_matrix.cols
                             << ", mat size: " << input_matrix.total() * input_matrix.elemSize()
                             << std::endl;

  return ErrorCodes::Success;
}

int
Context::GetControlInput(
    CustomGetNextInputFn_t input_fn, void* input_context,
    int32_t& start, int32_t& end, int32_t& ready, uint64_t& corrid)
{
  int err;

  // get the control inputs(START, END, READY, CORRID)
  // byte sizeでデータを取得するため、uint8のバッファに格納してから、各データの型に合わせて、変換する
  size_t batch1_control_input_size = GetDataTypeByteSize(TYPE_INT32);
  std::vector<uint8_t> start_buffer, end_buffer, ready_buffer, corrid_buffer;
  err = GetInputTensor(
      input_fn, input_context, "START", batch1_control_input_size,
      &start_buffer);
  if (err != ErrorCodes::Success) {
    return err;
  }

  err = GetInputTensor(
      input_fn, input_context, "END", batch1_control_input_size,
      &end_buffer);
  if (err != ErrorCodes::Success) {
    return err;
  }

  err = GetInputTensor(
      input_fn, input_context, "READY", batch1_control_input_size,
      &ready_buffer);
  if (err != ErrorCodes::Success) {
    return err;
  }

  batch1_control_input_size = GetDataTypeByteSize(TYPE_UINT64);
  err = GetInputTensor(
      input_fn, input_context, "CORRID", batch1_control_input_size,
      &corrid_buffer);
  if (err != ErrorCodes::Success) {
    return err;
  }

  // convert value from pointer
  start = *reinterpret_cast<int32_t*>(&start_buffer[0]);
  end = *reinterpret_cast<int32_t*>(&end_buffer[0]);
  ready = *reinterpret_cast<int32_t*>(&ready_buffer[0]);
  corrid = *reinterpret_cast<uint64_t*>(&corrid_buffer[0]);

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

int
Context::Output(
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
        LOG_INFO << INSTANCE_NAME << " error to allocate memory" << std::endl;
        return kOutputBuffer;
  }

  // copy data into output buffer
  if (obuffer != nullptr) {
      memcpy(obuffer, output.c_str(), output.size());
  }
  return ErrorCodes::Success;
}

std::string 
Context::CreateResultTreeStr(std::vector<std::vector<face_recognition_model::FaceData>> face_detect_datas)
{
  /*
    json format is following:
      "result":
      {
        "image_count": xx,
        "image_infos":
          [
            {
              "face_count": xx,
              "face_infos":
                [
                  {
                    "name": yy,
                    "box": 
                      {
                        "top": xx,
                        "bottom": xx,
                        "left": xx,
                        "right": xx
                      }
                  },
                  {
                    ...
                  },
                  ...
                ]
            },
            {
              ...
            },
          ]
      }
  */
  ptree pt;
  pt.put("result.image_count", face_detect_datas.size());

  ptree image_infos;
  for (auto image_data : face_detect_datas) {
    ptree image_info;
    image_info.put("face_count", image_data.size());

    ptree face_infos;
    for (auto face_data : image_data) {
      ptree face_info;
      face_info.put("name", face_data.Name());
      face_info.put("box.top", face_data.Rect().top());
      face_info.put("box.bottom", face_data.Rect().bottom());
      face_info.put("box.left", face_data.Rect().left());
      face_info.put("box.right", face_data.Rect().right());
      face_infos.push_back(std::make_pair("", face_info));
    }
    image_info.add_child("face_infos", face_infos);
    image_infos.push_back(std::make_pair("", image_info));
  }
  pt.add_child("image_infos", image_infos);

  // convert ptree to std::string
  std::ostringstream oss;
  write_json(oss, pt);
  std::string result = oss.str();
  oss.clear();
  return result;
}

}  // namespace face_recognition_model

// Creates a new face_recognition_model context instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
  face_recognition_model::Context* context =
      new face_recognition_model::Context(name, model_config, gpu_device);

  *instance = context;

  if (context == nullptr) {
    return ErrorCodes::CreationFailure;
  }

  return context->Init();
}

}}}  // namespace nvidia::inferenceserver::custom
