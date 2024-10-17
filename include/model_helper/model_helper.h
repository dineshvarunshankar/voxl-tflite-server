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

struct ModelHelperMeta
{
    std::unique_ptr<ModelHelper> helper;
    std::unique_ptr<PostprocessParams> params;

    ModelHelperMeta(std::unique_ptr<ModelHelper> h, std::unique_ptr<PostprocessParams> p)
        : helper(std::move(h)), params(std::move(p)) {}
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

    virtual bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params);
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

class ObjectDetectionModelHelper : public ModelHelper
{
private:
    std::vector<std::string> labels;
    size_t label_count;

public:
    ObjectDetectionModelHelper(char *model_file, char *labels_file,
                         DelegateOpt delegate_choice, bool _en_debug,
                         bool _en_timing, NormalizationType _do_normalize);

    bool postprocess(cv::Mat &output_image, double last_inference_time, void *input_params) override;
};

#endif // MODEL_HELPER_H
