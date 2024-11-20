#include "model_helper/fast_depth_model_helper.h"
#include "tensor_data.h"
#include "image_utils.h"

FastDepthModelHelper::FastDepthModelHelper(char *model_file, char *labels_file,
                                           DelegateOpt delegate_choice, bool _en_debug,
                                           bool _en_timing, NormalizationType _do_normalize)
    : ModelHelper(model_file, labels_file, delegate_choice, _en_debug, _en_timing, _do_normalize) {}

bool FastDepthModelHelper::postprocess(cv::Mat &output_image, double last_inference_time, void *input_params)
{
    printf("inside postprocess 1\n");
    FastDepthModelParams *params = static_cast<FastDepthModelParams*>(input_params);
    printf("inside postprocess 1.5\n");
    printf("params is null %d\n", params == nullptr);

    camera_image_metadata_t &meta = params->meta;
    printf("inside postprocess 2\n");
    start_time = rc_nanos_monotonic_time();

    TfLiteTensor *output_locations =
        interpreter->tensor(interpreter->outputs()[0]);
    float *depth = TensorData<float>(output_locations, 0);
    printf("inside postprocess 3\n");

    // actual depth image if desired
    cv::Mat depthImage(model_height, model_width, CV_32FC1, depth);
    printf("inside postprocess 4\n");

    // setup output metadata
    meta.height = model_height;
    meta.width = model_width;
    meta.size_bytes = meta.width * meta.height * 3;
    meta.stride = meta.width * 3;
    meta.format = IMAGE_FORMAT_RGB;
    printf("inside postprocess 5\n");

    // create a pretty colored depth image from the data
    double min_val, max_val;
    cv::Mat depthmap_visual;
    cv::minMaxLoc(depthImage, &min_val, &max_val);
    depthmap_visual = 255 * (depthImage - min_val) /
                      (max_val - min_val); // * 255 for "scaled" disparity
    depthmap_visual.convertTo(depthmap_visual, CV_8U);
    cv::applyColorMap(depthmap_visual, output_image, 4); // opencv COLORMAP_JET
    printf("inside postprocess 6\n");

    if (en_timing)
        total_postprocess_time +=
            ((rc_nanos_monotonic_time() - start_time) / 1000000.);
    printf("inside postprocess 7\n");

    draw_fps(output_image, last_inference_time, cv::Point(0, 0), 0.5, 2,
             cv::Scalar(0, 0, 0), cv::Scalar(180, 180, 180), true);
    printf("inside postprocess 8\n");

    delete params;

    return true;
}

bool FastDepthModelHelper::worker(cv::Mat &output_image, double last_inference_time, TFLiteMessage *new_frame, void *input_params)
{
    printf("fast depth model helper 1\n");
    if (new_frame == nullptr) {
        return false;
    }
    new_frame_metadata = new_frame->metadata;


    void* params;

    if (input_params == nullptr)
        params = new FastDepthModelParams(new_frame->metadata);
    else
        params = input_params;

    if (!postprocess(output_image, last_inference_time, params))
        return false;
    new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
    pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata,
                                   (char *)output_image.data);
    printf("fast depth model helper 2\n");
    return true;
}