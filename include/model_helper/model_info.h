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
    OBJECT_DETECT_MODEL,
    MONO_DEPTH_MODEL,
    SEGMENTATION_MODEL,
    CLASSIFICATION_MODEL,
    POSENET,
    YOLOV5,
    YOLOV8
};

enum ModelCategory
{
    OBJECT_DETECTION,
    CLASSIFICATION,
    SEGMENTATION,
    MONO_DEPTH,
    POSE
};

extern std::map<ModelName, ModelCategory> model_category_map;

class PostprocessParams
{
public:
    virtual ~PostprocessParams() = default; // Virtual destructor for proper cleanup
};

class ObjectDetectionModelParams : public PostprocessParams
{
public:
    std::vector<ai_detection_t> detections_vector;
};

#endif