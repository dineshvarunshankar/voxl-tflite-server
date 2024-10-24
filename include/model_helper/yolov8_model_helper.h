#ifndef YOLOV8_H
#define YOLOV8_H

#include "model_helper/model_helper.h"

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
    bool run_inference(cv::Mat preprocessed_image,
                        double *last_inference_time) override;

private:
    std::vector<std::string> labels;
    size_t label_count;
    const float model_score_threshold = 0.45;
    const float model_confidence_threshold = 0.25;
    const float model_nms_threshold = 0.5;
};

#endif