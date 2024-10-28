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
};

void *inference_worker(void *data);

#endif