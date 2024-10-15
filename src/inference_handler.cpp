#include "inference_handler.h"

// pass in the pointer to the subclass here
void *inference_worker(void *args)
{
    InferenceWorkerArgs *worker_args = static_cast<InferenceWorkerArgs *>(args);
    ModelHelper *model_helper = worker_args->model_helper;
    ModelName model_name = worker_args->model_name;

    // Set thread priority
    pid_t tid = syscall(SYS_gettid);
    int which = PRIO_PROCESS;
    int nice = -15;
    setpriority(which, tid, nice);

    // if we can, chuck this thread onto the big cache cores
#ifdef BUILD_QRB5165

    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();

    /* Set affinity mask to include CPUs 7 only */
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    CPU_SET(5, &cpuset);
    CPU_SET(4, &cpuset);

    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
    {
        perror("pthread_setaffinity_np");
    }

    /* Check the actual affinity mask assigned to the thread */
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset))
    {
        perror("pthread_getaffinity_np");
    }
    printf("Camera processing thread is now locked to the following cores:");
    for (int j = 0; j < CPU_SETSIZE; j++)
    {
        if (CPU_ISSET(j, &cpuset))
            printf(" %d", j);
    }
    printf("\n");

#endif

    // keep track of where we are in terms of processing the tflite camera queue
    int queue_index = 0;

    while (main_running)
    {

        if (queue_index == model_helper->camera_queue.insert_idx)
        {
            std::unique_lock<std::mutex> lock(model_helper->cond_mutex);
            model_helper->cond_var.wait(lock);
            continue;
        }
        // grab the frame and bump our queue index, making sure its within queue
        // size
        TFLiteMessage *new_frame = &model_helper->camera_queue.queue[queue_index];
        queue_index = ((queue_index + 1) % QUEUE_SIZE);

        cv::Mat preprocessed_image, output_image;

        if (!model_helper->preprocess_image(new_frame->metadata, (char *)new_frame->image_pixels, preprocessed_image, output_image))
            continue;

        int new_format = new_frame->metadata.format;

        double last_inference_time = 0;

        if (!model_helper->run_inference(preprocessed_image, &last_inference_time))
            continue;

        new_frame->metadata.format = new_format;

        // Now creating the post process input param package
        switch (model_name)
        {
        case OBJECT_DETECT_MODEL:
        {
            std::vector<ai_detection_t> detections;
            auto params = std::make_unique<ObjectDetectionParams>(detections);
            if (!model_helper->postprocess(output_image, last_inference_time, params.get()))
                continue;

            if (!detections.empty())
            {
                for (unsigned int i = 0; i < detections.size(); i++)
                {
                    pipe_server_write(DETECTION_CH, (char *)&detections[i],
                                      sizeof(ai_detection_t));
                }
            }
            new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
            pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata,
                                           (char *)output_image.data);

            break;
        }

        default:
        {
            // Handle unsupported model types or an error case
            fprintf(stderr, "Error: Unsupported model type.\n");
            break;
        }
        }
    }

    delete worker_args;
    return nullptr;
}

void initialize_inference_settings(char *model, char *delegate, int *tensor_offset, ModelName *model_name, NormalizationType *norm_type, DelegateOpt *opt)
{

    // Set delegate
    *opt = GPU; // default for MAI models
    if (!strcmp(delegate, "cpu"))
        *opt = XNNPACK;
    else if (!strcmp(delegate, "nnapi"))
        *opt = NNAPI;

    // set model type
    if (!strcmp(model, "/usr/bin/dnn/ssdlite_mobilenet_v2_coco.tflite"))
    {
        *model_name = OBJECT_DETECT_MODEL;
        *norm_type =
            PIXEL_MEAN; // funky for mobilenet, doesn't like hard division
    }
    else if (!strcmp(model, "/usr/bin/dnn/mobilenetv1_nnapi_quant.tflite"))
    {
        *model_name = OBJECT_DETECT_MODEL;
    }
    else if (!strcmp(model, "/usr/bin/dnn/fastdepth_float16_quant.tflite"))
    {
        *model_name = MONO_DEPTH_MODEL;
        *norm_type = HARD_DIVISION;
    }
    else if (!strcmp(model,
                     "/usr/bin/dnn/"
                     "edgetpu_deeplab_321_os32_float16_quant.tflite"))
    {
        *model_name = SEGMENTATION_MODEL;
        *norm_type = NONE;
    }
    else if (!strcmp(model,
                     "/usr/bin/dnn/"
                     "lite-model_efficientnet_lite4_uint8_2.tflite"))
    {
        *model_name = CLASSIFICATION_MODEL;
        *norm_type = PIXEL_MEAN;
    }
    else if (!strcmp(model,
                     "/usr/bin/dnn/mobilenetv1_nnapi_classifier.tflite"))
    {
        *model_name = CLASSIFICATION_MODEL;
        *norm_type = PIXEL_MEAN;
        // mobilenet special for background class!!!
        *tensor_offset = 1;
    }
    else if (!strcmp(model,
                     "/usr/bin/dnn/"
                     "lite-model_movenet_singlepose_lightning_tflite_float16_"
                     "4.tflite"))
    {
        *model_name = POSENET;
        *norm_type = NONE;
    }
    else if (!strcmp(model, "/usr/bin/dnn/yolov5_float16_quant.tflite"))
    {
        *model_name = YOLOV5;
        *norm_type = HARD_DIVISION;
    }
    else if (!strcmp(model, "/usr/bin/dnn/yolov8n_float16.tflite"))
    {
        *model_name = YOLOV8;
        *norm_type = HARD_DIVISION;
    }
    else
    {
        fprintf(stderr,
                "WARNING: Unknown model type provided! Defaulting post-process "
                "to object detection.\n");
        *model_name = OBJECT_DETECT_MODEL;
    }
}