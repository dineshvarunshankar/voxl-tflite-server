#ifndef MODEL_INFO_H
#define MODEL_INFO_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <modal_pipe.h>
#include <stdint.h>
#include <memory>
#include <opencv2/imgproc/types_c.h>
#include "ai_detection.h"

// ModelName refers to the name of the model
// Any name with *_MODEL is a generic for that model category
// (as defined in the map)
enum ModelName
{
    MOBILE_NET,
    FAST_DEPTH,
    DEEPLAB,
    EFFICIENT_NET,
    POSENET,
    YOLOV5,
    YOLOV8,
    PLACEHOLDER
};

// The category enum is a kinda redundant, might get rid 
// of it all together
enum ModelCategory
{
    OBJECT_DETECTION,
    CLASSIFICATION,
    SEGMENTATION,
    MONO_DEPTH,
    POSE
};

extern std::map<ModelName, ModelCategory> model_category_map;

class ObjectDetectionModelParams
{
public:
    std::vector<ai_detection_t> detections_vector;

    // Constructor to initialize detections_vector
    ObjectDetectionModelParams(const std::vector<ai_detection_t> &detections)
        : detections_vector(detections) {}
};

class ClassificationModelParams
{
public:
    int tensor_offset;

    ClassificationModelParams(int tensor_offset)
        : tensor_offset(tensor_offset) {}
};


#endif