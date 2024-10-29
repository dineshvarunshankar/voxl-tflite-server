#include "inference_handler.h"

// pass in the pointer to the subclass here
void *inference_worker(void *args)
{

    InferenceWorkerArgs *worker_args = static_cast<InferenceWorkerArgs *>(args);
    ModelHelper *model_helper = worker_args->model_helper;

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

        model_helper->preprocessed_image = &preprocessed_image;
        if (!model_helper->preprocess_image(new_frame->metadata, (char *)new_frame->image_pixels, preprocessed_image, output_image))
            continue;

        int new_format = new_frame->metadata.format;

        double last_inference_time = 0;

        if (!model_helper->run_inference(preprocessed_image, &last_inference_time))
        {

            continue;
        }

        new_frame->metadata.format = new_format;

        // sets up post processing and related operations 
        model_helper->worker(output_image, last_inference_time, new_frame);
    }

    delete worker_args;
    return nullptr;
}