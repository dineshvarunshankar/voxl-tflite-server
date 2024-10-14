#include <vector>
#include "ai_detection.h"

struct ObjectDetectionParams
{
    std::vector<ai_detection_t> detections_vector; 
    double last_inference_time;

    // Constructor to initialize members
    ObjectDetectionParams(const std::vector<ai_detection_t> &detections, double inference_time)
        : detections_vector(detections), last_inference_time(inference_time) {}
};


