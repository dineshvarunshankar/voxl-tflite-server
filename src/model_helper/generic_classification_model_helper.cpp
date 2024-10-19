#include "model_helper/model_helper.h"
#include "tensor_data.h"
#include "image_utils.h"

GenericClassificationModelHelper::GenericClassificationModelHelper(char *model_file, char *labels_file,
                                                                   DelegateOpt delegate_choice, bool _en_debug,
                                                                   bool _en_timing, NormalizationType _do_normalize)
    : ModelHelper(model_file, labels_file, delegate_choice, _en_debug, _en_timing, _do_normalize) {}

bool GenericClassificationModelHelper::postprocess(cv::Mat &output_image, double last_inference_time, void *input_params)
{
    ClassificationModelParams *params = static_cast<ClassificationModelParams *>(input_params);

    int tensor_offset = params->tensor_offset;

    int num_of_classes = 1000;

    start_time = rc_nanos_monotonic_time();

    static std::vector<std::string> labels;
    static size_t label_count;

    if (labels.empty())
    {
        if (ReadLabelsFile(labels_location, &labels, &label_count) !=
            kTfLiteOk)
        {
            fprintf(stderr, "ERROR: Unable to read labels file\n");
            return false;
        }
    }

    TfLiteTensor *output_locations =
        interpreter->tensor(interpreter->outputs()[0]);
    uint8_t *confidence_tensor = TensorData<uint8_t>(output_locations, 0);

    std::vector<uint8_t> confidences;
    confidences.assign(
        confidence_tensor + tensor_offset,
        confidence_tensor + num_of_classes + tensor_offset);

    uint8_t best_prob =
        *std::max_element(confidences.begin(), confidences.end());
    int best_class = std::max_element(confidences.begin(), confidences.end()) -
                     confidences.begin();

    fprintf(stderr, "class: %s, prob: %d\n", labels[best_class].c_str(),
            best_prob);
    cv::putText(output_image, labels[best_class],
                cv::Point(input_width / 3, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 1);

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2,
             cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);

    if (en_timing)
        total_postprocess_time +=
            ((rc_nanos_monotonic_time() - start_time) / 1000000.);

    return true;
}