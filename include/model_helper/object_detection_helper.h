#ifndef OBJECT_DETECTION_HELPER_H
#define OBJECT_DETECTION_HELPER_H

#include "model_helper/model_helper.h"
#include "model_helper/input_params.h"

class ObjectDetectionModel : public ModelHelper
{
private:
    std::vector<std::string> labels;
    size_t label_count;

public:
    ObjectDetectionModel(char *model_file, char *labels_file,
                         DelegateOpt delegate_choice, bool _en_debug,
                         bool _en_timing, NormalizationType _do_normalize);

    // Virtual function override for post-processing
    bool postprocess(cv::Mat &output_image, void* input_params) override;
};

#endif // OBJECT_DETECTION_HELPER_H