#include <string>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <fstream>
#include <sstream>

// custom header
#include "custom_backend/face_recognition_model/face_wrapper.h"

// dlib
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
using namespace dlib;

namespace nvidia { namespace inferenceserver { namespace custom {
namespace face_recognition_model {

FaceWrapper::FaceWrapper() {
}

FaceWrapper::~FaceWrapper() {
}

int
FaceWrapper::Initialize(float distance_threshold) {
  // set parameter
  face_detector = get_frontal_face_detector();
  this->distance_threshold = distance_threshold;

  // model datas are copied on models folder (TODO : enable to set the path of model)
  deserialize("/models/face_recognition_model/dlib_face_recognition_resnet_model_v1.dat") >> net;
  deserialize("/models/face_recognition_model/shape_predictor_5_face_landmarks.dat") >> sp;

  // register face database (TODO : データセットのパスをconfigから変更できるようにする)
  int register_count = registerFaceDatasFromCsv("/dataset/face/face.csv", "/dataset/face/image/");
  std::cout << LOG_PREFIX << " initialize : register face data count is " << register_count << std::endl;

  // print DLIB_USE_CUDA if needed
  if (kDebugMode) {
#ifdef DLIB_USE_CUDA
    std::cout << LOG_PREFIX << " DLIB mode is GPU" << std::endl;
#else
    std::cout << LOG_PREFIX << " DLIB mode is CPU" << std::endl;
#endif
  }

  return ErrorCodes::Success;
}

std::vector<face_recognition_model::FaceData>
FaceWrapper::ExtractFaceData(cv::Mat cv_img)
{
  std::vector<face_recognition_model::FaceData> face_datas;

  // convert dlib image data from opencv image data
  auto dlib_cv_img = dlib::cv_image<dlib::rgb_pixel>(cv_img);

  // extract face rects
  std::vector<dlib::rectangle> dets = extractFaceRect(dlib_cv_img);

  // check count of dets
  if (dets.empty()) {
    std::cout << LOG_PREFIX << " face data is not found" << std::endl;
    return face_datas;
  }

  // create face datas
  auto face_datas_in_db = GetFaceDatasFromDB();
  for (auto det : dets) {
    // vectorize the face data
    auto face_descriptor = calcFaceDescriptor(dlib_cv_img, det);

    // judge the similar face
    auto face_name = predictSimilarFace(face_descriptor, face_datas_in_db, distance_threshold);

    // create face data
    face_recognition_model::FaceData data(det, face_descriptor);
    data.SetName(face_name);
    face_datas.push_back(data);
  }

  return face_datas;
}

std::vector<face_recognition_model::FaceData>
FaceWrapper::GetFaceDatasFromDB()
{
  return face_db;
}

std::vector<dlib::rectangle> 
FaceWrapper::extractFaceRect(dlib::cv_image<dlib::rgb_pixel> dlib_cv_img)
{
  return face_detector(dlib_cv_img);
}

matrix<rgb_pixel>
FaceWrapper::extractFaceChip(dlib::cv_image<dlib::rgb_pixel> dlib_cv_img, dlib::rectangle face_rect)
{
  auto shape = sp(dlib_cv_img, face_rect);
  matrix<rgb_pixel> face_chip;
  extract_image_chip(dlib_cv_img, get_face_chip_details(shape,150,0.25), face_chip);
  return face_chip;
}

matrix<float,0,1> 
FaceWrapper::calcFaceDescriptor(dlib::cv_image<dlib::rgb_pixel> dlib_cv_img, dlib::rectangle face_rect)
{
  // extract face chips (ex. nose, mouth..)
  auto face_chip = extractFaceChip(dlib_cv_img, face_rect);
  std::vector<matrix<rgb_pixel>> faces;
  faces.push_back(face_chip);

  // vectorize the face data (TODO : まとめて画像を入れた方が速度あがるかも？)
  std::vector<matrix<float,0,1>> face_descriptors = net(faces);
  return face_descriptors[0];
}

std::string 
FaceWrapper::predictSimilarFace(matrix<float,0,1> target_descriptor, std::vector<face_recognition_model::FaceData> reference_face_datas, float distance_threshold)
{
  if (reference_face_datas.size() == 0) {
    return FACE_TYPE_UNKNOWN;
  }

  // TODO : 顔距離の閾値を設定し、そのうちで最も近い顔データを採用（将来的にはSVMなどの分類器を使った方が良い）
  // calculate the face distance between target face data and face datas of database
  std::map<float, std::string> face_distance_infos;
  for (auto ref_face_data : reference_face_datas) {
    float distance = length(target_descriptor - ref_face_data.GetFaceDescriptor());
    face_distance_infos[distance] = ref_face_data.Name();
  }

  // check threshold
  if (face_distance_infos.begin()->first < distance_threshold) {
    return face_distance_infos.begin()->second;
  }
  return FACE_TYPE_UNKNOWN;
}

int 
FaceWrapper::registerFaceDatas(std::vector<cv::Mat> cv_imgs, std::vector<std::string> face_names)
{
  // error check
  if (cv_imgs.size() != face_names.size()) {
    return kFaceRegisterError;
  }

  // get face vector
  for (std::size_t i = 0; i < cv_imgs.size(); i++) {
    // create and register face data
    face_recognition_model::FaceData data;
    auto error_code = createFaceData(cv_imgs[i], face_names[i], data);
    if (error_code != ErrorCodes::Success) {
      continue;
    }
    face_db.push_back(data);
  }
  return ErrorCodes::Success;
}

int
FaceWrapper::createFaceData(cv::Mat cv_img, std::string face_name, face_recognition_model::FaceData& face_data)
{
  auto dlib_cv_img = dlib::cv_image<dlib::rgb_pixel>(cv_img);
  std::vector<dlib::rectangle> dets = extractFaceRect(dlib_cv_img);

  // check error
  if (dets.empty()) {
    // if face rect is not found, this face data is ignored.
    std::cout << LOG_PREFIX << " face data is not found : " << face_name << std::endl;
    return kFaceNotFound;
  } else if (dets.size() > 1) {
    // if multi face data is detected in one image data, this face datas are ignored.
    std::cout << LOG_PREFIX << " multi face datas are detected! this image is ignored : " << face_name << std::endl;
    return kFaceMultiple;
  }

  // vectorize the face data
  dlib::rectangle det = dets[0];
  auto face_descriptor = calcFaceDescriptor(dlib_cv_img, det);

  // create face data
  face_recognition_model::FaceData data(det, face_descriptor);
  data.SetName(face_name);
  face_data = data;
  return ErrorCodes::Success;
}

int 
FaceWrapper::registerFaceDatasFromCsv(std::string csv_path, std::string root_folder_path)
{
  int register_count = 0;
  std::ifstream ifs(csv_path);
  std::string line;
  while (getline(ifs, line)) {
    auto texts = split_text(line, ',');

    // texts have the following contents (each line has 2 columns)
    // [0] -> image file name
    // [1] -> face name
    if (texts.size() != 2) {
      // TODO : display error log
      continue;
    }
    std::string csv_file_name = texts[0];
    std::string face_name = texts[1];

    // create opencv matrix
    auto path = root_folder_path + csv_file_name;
    cv::Mat cv_img = cv::imread(path, 1);

    // create face data
    face_recognition_model::FaceData data;
    auto error_code = createFaceData(cv_img, face_name, data);

    // register face data if successed
    if (error_code != ErrorCodes::Success) {
      // TODO : display error log
      continue;
    }
    face_db.push_back(data);
    register_count++;
  }

  return register_count;
}

std::vector<std::string> 
FaceWrapper::split_text(const std::string s, char delim)
{
  std::vector<std::string> elems;
  std::string item;
  for (char ch: s) {
      if (ch == delim) {
          if (!item.empty())
              elems.push_back(item);
          item.clear();
      }
      else {
          item += ch;
      }
  }
  if (!item.empty())
      elems.push_back(item);
  return elems;
}

}  // namespace face_recognition_model
}}}  // namespace nvidia::inferenceserver::custom
