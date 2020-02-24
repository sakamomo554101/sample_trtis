// trtis
#include "src/custom/sdk/custom_instance.h"

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// dlib
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
using namespace dlib;

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

#define FACE_TYPE_UNKNOWN "unknown"

namespace nvidia { namespace inferenceserver { namespace custom {
namespace face_recognition_model {

// this class contains the face area and name
class FaceData {
  public:
    FaceData()
    {
      // TODO : this constructor should be removed bacause some values of this class are not initialized.
      this->name = FACE_TYPE_UNKNOWN;
    }
    FaceData(const dlib::rectangle rect, const matrix<float,0,1> face_descriptor) 
    {
      this->rect = rect;
      this->face_descriptor = face_descriptor;
      this->name = FACE_TYPE_UNKNOWN;
    }
    ~FaceData() {}

    dlib::rectangle Rect()
    {
      return this->rect;
    }

    std::string Name()
    {
      return this->name;
    }

    void SetName(std::string name)
    {
      this->name = name;
    }

    matrix<float,0,1> GetFaceDescriptor()
    {
      return this->face_descriptor;
    }

  private:
    dlib::rectangle rect;
    matrix<float,0,1> face_descriptor;
    std::string name;
};

class FaceWrapper {
  public:
    FaceWrapper();
    ~FaceWrapper();

    int Initialize(float distance_threshold = 0.6);

    std::vector<face_recognition_model::FaceData> ExtractFaceData(cv::Mat cv_img);
    std::vector<face_recognition_model::FaceData> GetFaceDatasFromDB();

    // error codes
    const int kFaceRegisterError = -1;
    const int kFaceNotFound = -2;
    const int kFaceMultiple = -3;

  private:
    const std::string LOG_PREFIX = "[FaceWrapper]";

    // face detector
    dlib::frontal_face_detector face_detector;

    // shape predictor
    dlib::shape_predictor sp;

    // vectorizer of face data
    anet_type net;

    // face distance threshold
    float distance_threshold;

    // TODO : データベースは別で作った方が良い
    std::vector<face_recognition_model::FaceData> face_db;

    // private functions
    std::vector<dlib::rectangle> extractFaceRect(dlib::cv_image<dlib::rgb_pixel> dlib_cv_img);
    matrix<rgb_pixel> extractFaceChip(dlib::cv_image<dlib::rgb_pixel> dlib_cv_img, dlib::rectangle face_rect);
    matrix<float,0,1> calcFaceDescriptor(dlib::cv_image<dlib::rgb_pixel> dlib_cv_img, dlib::rectangle face_rect);
    std::string predictSimilarFace(matrix<float,0,1> target_descriptor, std::vector<face_recognition_model::FaceData> reference_face_datas, float distance_threshold);
    int registerFaceDatas(std::vector<cv::Mat> cv_imgs, std::vector<std::string> face_names);
    int createFaceData(cv::Mat cv_img, std::string face_name, face_recognition_model::FaceData& face_data);
    int registerFaceDatasFromCsv(std::string csv_path, std::string root_folder_path);

    // util
    std::vector<std::string> split_text(const std::string s, char delim);
};

}  // namespace face_recognition_model
}}}  // namespace nvidia::inferenceserver::custom

