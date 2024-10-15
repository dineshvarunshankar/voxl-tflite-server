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
#include "model_helper/input_params.h"

#define DETECTION_CH 1
#define IMAGE_CH 0

struct InferenceWorkerArgs
{
    ModelHelper *model_helper;
    ModelName model_name;
};

void initialize_inference_settings(char *model, char *delegate, int *tensor_offset, ModelName *model_type, NormalizationType *norm_type, DelegateOpt *opt);

void *inference_worker(void *data);
