#ifndef MODEL_HELPER_H
#define MODEL_HELPER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <modal_pipe.h>
#include <stdint.h>
#include <memory>
#include <opencv2/imgproc/types_c.h>
#include <mutex>
#include <condition_variable>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "ai_detection.h"
#include "utils.h"
#include "resize.h"

#ifdef BUILD_QRB5165
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif

#include "config_file.h"
#include "tensor_data.h"
#include "model_info.h"

#define MAX_IMAGE_SIZE 12441600
#define QUEUE_SIZE 24 // max messages to be stored in queue

#define NORMALIZATION_CONST 255.0f
#define PIXEL_MEAN_GUESS 127.0f

enum DelegateOpt
{
    XNNPACK,
    GPU,
    NNAPI
};

enum NormalizationType
{
    NONE,
    PIXEL_MEAN,
    HARD_DIVISION
};

struct TFLiteMessage
{
    camera_image_metadata_t metadata;     // image metadata information
    uint8_t image_pixels[MAX_IMAGE_SIZE]; // image pixels
};

struct TFLiteCamQueue
{
    TFLiteMessage queue[QUEUE_SIZE]; // camera frame queue
    int insert_idx = 0;              // next element insert location (between 0 - QUEUE_SIZE)
};

class ModelHelper
{
protected:
    // holders for model specific data
    int model_width;
    int model_height;
    int model_channels;

    // cam properties
    int input_width;
    int input_height;

    // labels
    char *labels_location;

    // delegate ptrs
    DelegateOpt hardware_selection;
    TfLiteDelegate *gpu_delegate;
#ifdef BUILD_QRB5165
    TfLiteDelegate *xnnpack_delegate;
    tflite::StatefulNnApiDelegate *nnapi_delegate;
#endif

    // only used if running an object detection model
    // ai_detection_t detection_data;
    // char *labels_location;
    // char *model;

    bool en_debug;
    bool en_timing;
    NormalizationType do_normalize;

    // timing variables
    float total_preprocess_time = 0;
    float total_inference_time = 0;
    float total_postprocess_time = 0;
    uint64_t start_time = 0;
    int num_frames_processed = 0;

    // tflite
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // mcv resize vars
    uint8_t *resize_output;
    undistort_map_t map;

public:
    ModelHelper(char *model_file, char *labels_file,
                DelegateOpt delegate_choice, bool _en_debug,
                bool _en_timing, NormalizationType _do_normalize);

    // preprocess method, common across most base classes
    virtual bool preprocess_image(camera_image_metadata_t &meta,
                                  char *frame, cv::Mat &preprocessed_image,
                                  cv::Mat &output_image);

    virtual bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params = nullptr) = 0;
    virtual bool run_inference(cv::Mat preprocessed_image,
                               double *last_inference_time);

    void print_summary_stats();

    virtual ~ModelHelper() = default;

    std::string cam_name;
    pthread_t thread;                 // model thread handle
    std::mutex cond_mutex;            // mutex
    std::condition_variable cond_var; // condition variable

    TFLiteCamQueue camera_queue; // camera message queue for the thread

protected:
    // Function to setup the delegate based on selection
    void setupDelegate(DelegateOpt delegate_choice);
};

ModelHelper *create_model_helper(ModelName model_name,
                                 DelegateOpt opt_,
                                 NormalizationType do_normalize);

class GenericObjectDetectionModelHelper : public ModelHelper
{
private:
    std::vector<std::string> labels;
    size_t label_count;

public:
    GenericObjectDetectionModelHelper(char *model_file, char *labels_file,
                                      DelegateOpt delegate_choice, bool _en_debug,
                                      bool _en_timing, NormalizationType _do_normalize);

    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;
};

class GenericClassificationModelHelper : public ModelHelper
{

public:
    GenericClassificationModelHelper(char *model_file, char *labels_file,
                                     DelegateOpt delegate_choice, bool _en_debug,
                                     bool _en_timing, NormalizationType _do_normalize);

    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;

private:
    std::vector<std::string> labels;
    size_t label_count;
};

class PoseNetModelHelper : public ModelHelper
{
public:
    PoseNetModelHelper(char *model_file, char *labels_file,
                       DelegateOpt delegate_choice, bool _en_debug,
                       bool _en_timing, NormalizationType _do_normalize);
    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;
};

class FastDepthModelHelper : public ModelHelper
{
public:
    FastDepthModelHelper(char *model_file, char *labels_file,
                         DelegateOpt delegate_choice, bool _en_debug,
                         bool _en_timing, NormalizationType _do_normalize);
    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;
};

class DeepLabModelHelper : public ModelHelper
{
public:
    DeepLabModelHelper(char *model_file, char *labels_file,
                       DelegateOpt delegate_choice, bool _en_debug,
                       bool _en_timing, NormalizationType _do_normalize);
    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;

private:
    static constexpr int right_pixel_border = 110;
    std::vector<std::string> labels;
    size_t label_count;
};

class YoloV5ModelHelper : public ModelHelper
{
public:
    YoloV5ModelHelper(char *model_file, char *labels_file,
                      DelegateOpt delegate_choice, bool _en_debug,
                      bool _en_timing, NormalizationType _do_normalize);
    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;

private:
    std::vector<std::string> labels;
    size_t label_count;
    // straight up stolen from
    // https://github.com/iwatake2222/play_with_tflite/blob/master/pj_tflite_det_yolov5/image_processor/detection_engine.cpp
    static constexpr int32_t kGridScaleList[] = {8, 16, 32};
    static constexpr int32_t kGridChannel = 3;
    static constexpr float threshold_box_confidence_ =
        0.40; // not sure if this is too low or high yet
    static constexpr float threshold_class_confidence_ =
        0.20; // not sure if this is too low or high yet
    static constexpr float threshold_nms_iou_ =
        0.50; // not sure if this is too low or high yet

    int32_t kElementNumOfAnchor;
    int32_t kNumberOfClass;

    struct b_box
    {
        int32_t class_id;
        std::string label;
        float class_conf;
        float detection_conf;
        float score;
        int32_t x;
        int32_t y;
        int32_t w;
        int32_t h;
    };

    void get_bbox(const float *data, float scale_x, float scale_y,
                  int32_t grid_w, int32_t grid_h, int number_of_classes,
                  std::vector<b_box> &bbox_list);
    void nms(std::vector<b_box> &bbox_list,
             std::vector<b_box> &bbox_nms_list, float threshold_nms_iou,
             bool check_class_id);
    float calc_iou(const b_box &obj0, const b_box &obj1);
};

class YoloV8ModelHelper : public ModelHelper
{
public:
    YoloV8ModelHelper(char *model_file, char *labels_file,
                      DelegateOpt delegate_choice, bool _en_debug,
                      bool _en_timing, NormalizationType _do_normalize);
    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;
    bool preprocess_image(camera_image_metadata_t &meta,
                          char *frame, cv::Mat &preprocessed_image,
                          cv::Mat &output_image) override;

private:
    std::vector<std::string> labels;
    size_t label_count;
    static constexpr float model_score_threshold = 0.45;
    static constexpr float model_confidence_threshold = 0.25;
    static constexpr float model_nms_threshold = 0.5;
};

#endif // MODEL_HELPER_H
