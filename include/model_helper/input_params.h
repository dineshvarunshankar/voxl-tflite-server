#include <vector>
#include "ai_detection.h"

struct ObjectDetectionParams
{
    std::vector<ai_detection_t> detections_vector; 

    ObjectDetectionParams(const std::vector<ai_detection_t> &detections)
        : detections_vector(detections) {}
};


