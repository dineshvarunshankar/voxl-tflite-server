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

#define DETECTION_CH 1
#define IMAGE_CH 0

struct InferenceWorkerArgs
{
    ModelHelper *model_helper;
    ModelName model_name;
    ModelCategory model_category;
};

void set_delegate(DelegateOpt *opt);

void initialize_model_settings(char *model, char *delegate, ModelName *model_type, ModelCategory *model_category, NormalizationType *norm_type, bool* custom_post);

void *inference_worker(void *data);

#endif