#include "inference_handler.h"

// pass in the pointer to the subclass here
void *inference_worker(void *args)
{


    InferenceWorkerArgs *worker_args = static_cast<InferenceWorkerArgs *>(args);
    ModelHelper *model_helper = worker_args->model_helper;
    ModelName model_name = worker_args->model_name;
    ModelCategory model_category = worker_args->model_category;

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

        if (!model_helper->run_inference(preprocessed_image, &last_inference_time)) {

            continue;
        }

        new_frame->metadata.format = new_format;

        // Now creating the post process input param package
        switch (model_name)
        {
        case MOBILE_NET:
        {
            if (model_category == OBJECT_DETECTION)
            {
                std::vector<ai_detection_t> detections;
                if (!generic_object_detection_worker(model_helper, output_image, last_inference_time, detections, new_frame))
                    continue;
            }

            else if (model_category == CLASSIFICATION)
            {
                int tensor_offset = 1;
                if (!generic_classification_worker(model_helper, output_image, last_inference_time, tensor_offset, new_frame))
                    continue;
            }

            // Handle other categories if any
            break;
        }

        case POSENET:
        {
            // potentially redundant checks but kept them in anyway
            if (model_category == POSE)
            {
                if (!generic_pose_worker(model_helper, output_image, last_inference_time, new_frame))
                    continue;
            }
            // Handle other categories if any
            break;
        }

        case FAST_DEPTH:
        {
            if (model_category == MONO_DEPTH)
            {
                if (!fast_depth_worker(model_helper, output_image, last_inference_time, new_frame))
                    continue;
            }
            // Handle other categories if any
            break;
        }

        case DEEPLAB:
        {
            if (model_category == SEGMENTATION)
            {
                if (!deep_lab_worker(model_helper, preprocessed_image, last_inference_time, new_frame))
                    continue;
            }
            // Handle other categories if any
            break;
        }

        case EFFICIENT_NET:
        {
            if (model_category == CLASSIFICATION)
            {
                int tensor_offset = 0;
                if (!generic_classification_worker(model_helper, output_image, last_inference_time, tensor_offset, new_frame))
                    continue;
            }
            // Handle other categories if any
            break;
        }

        case YOLOV5:
        {
            if (model_category == OBJECT_DETECTION)
            {
                std::vector<ai_detection_t> detections;
                if (!generic_object_detection_worker(model_helper, output_image, last_inference_time, detections, new_frame))
                    continue;
            }
            // Handle other categories if any
            break;
        }

        case YOLOV8:
        {
            if (model_category == OBJECT_DETECTION)
            {
                std::vector<ai_detection_t> detections;
                if (!generic_object_detection_worker(model_helper, output_image, last_inference_time, detections, new_frame))
                    continue;
            }
            // Handle other categories if any
            break;
        }

        default:
        {
            // Handle unsupported model types or an error case
            fprintf(stderr, "Error: Unsupported model.\n");
            break;
        }
        }
    }

    delete worker_args;
    return nullptr;
}

bool generic_object_detection_worker(ModelHelper *model_helper,
                                            cv::Mat &output_image,
                                            double last_inference_time,
                                            std::vector<ai_detection_t> &detections,
                                            TFLiteMessage *new_frame)
{

    auto params = std::make_unique<GenericObjectDetectionModelParams>(detections);

    if (!model_helper->postprocess(output_image, last_inference_time, params.get()))
        return false;

    if (!detections.empty())
    {
        for (unsigned int i = 0; i < detections.size(); i++)
        {
            pipe_server_write(DETECTION_CH, (char *)&detections[i], sizeof(ai_detection_t));
        }
    }

    new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
    pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata, (char *)output_image.data);

    return true;
}

bool generic_classification_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, int tensor_offset, TFLiteMessage *new_frame)
{
    auto params = std::make_unique<ClassificationModelParams>(tensor_offset);

    if (!model_helper->postprocess(output_image, last_inference_time, params.get()))
        return false;

    new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
    pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata,
                                   (char *)output_image.data);
    return true;
}

bool generic_pose_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, TFLiteMessage *new_frame)
{
    if (!model_helper->postprocess(output_image, last_inference_time))
        return false;
    new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
    pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata,
                                   (char *)output_image.data);
    return true;
}

bool fast_depth_worker(ModelHelper *model_helper, cv::Mat &output_image, double last_inference_time, TFLiteMessage *new_frame)
{
    auto params = std::make_unique<FastDepthModelParams>(new_frame->metadata);

    if (!model_helper->postprocess(output_image, last_inference_time, params.get()))
        return false;
    new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
    pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata,
                                   (char *)output_image.data);
    return true;
}

bool deep_lab_worker(ModelHelper *model_helper, cv::Mat &preprocessed_image, double last_inference_time, TFLiteMessage *new_frame)
{
    auto params = std::make_unique<FastDepthModelParams>(new_frame->metadata);

    // Segmentation is a special case here
    // instead of passing the full dimension "output_image", we pass the
    // preprocessed_image back then, the model output and overlay image
    // are the same dims so we can easily blend the two
    if (!model_helper->postprocess(preprocessed_image, last_inference_time, params.get()))
        return false;
    new_frame->metadata.timestamp_ns = rc_nanos_monotonic_time();
    pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata,
                                   (char *)preprocessed_image.data);
    return true;
}
