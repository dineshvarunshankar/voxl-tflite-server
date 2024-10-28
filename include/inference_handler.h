#ifndef INFERENCE_HANDLER_H
#define INFERENCE_HANDLER_H

#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <modal_pipe.h>

#include "model_helper/model_helper.h"
#include "model_helper/model_info.h"

struct InferenceWorkerArgs
{
    ModelHelper *model_helper;
    ModelName model_name;
    ModelCategory model_category;
};

bool generic_object_detection_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, std::vector<ai_detection_t> &detections, TFLiteMessage *new_frame);
bool generic_classification_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, int tensor_offset, TFLiteMessage *new_frame);
bool generic_pose_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, TFLiteMessage *new_frame);
bool deep_lab_worker(ModelHelper *model_helper, cv::Mat &preprocessed_image, double last_inference_time, TFLiteMessage *new_frame);
bool fast_depth_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, TFLiteMessage *new_frame);

void *inference_worker(void *data);

#endif