#ifndef Face_hpp
#define Face_hpp
#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include "Matrix.h"

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, KEY_TAG, __VA_ARGS__)
#define KEY_TAG "KEYPOINT"
using namespace  std;

typedef struct FaceInfo{
    float x_min;
    float y_min;
    float x_max;
    float y_max;
    float score;
}FaceInfo;


class Face {
public:
    Face(std::string model_path);
    static float IOU(FaceInfo boxes_one,FaceInfo boxes_two);
    static bool sort_score(FaceInfo boxes_one,FaceInfo boxes_two);
    static std::vector<FaceInfo> NMS(std::vector<FaceInfo> boxes,float threshold);
    float* detection(unsigned char *image_data, int width, int height, int channel);

private:
    std::shared_ptr<MNN::Interpreter>face_interpreter = nullptr;
    MNN::Session *face_session = nullptr;
    MNN::CV::ImageProcess::Config image_config;
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;

    int WIDTH = 128;
    int HEIGHT = 128;
    int CHANNELS = 3;
    int THREADS = 2;
    int OUTPUT_NUM = 960;
    float X_SCALE = 10.0;
    float Y_SCALE = 10.0;
    float H_SCALE = 5.0;
    float W_SCALE = 5.0;
    float score_threshold = 0.5f;
    float nms_threshold = 0.45f;
    const float MEAN[3] = {0.0f,0.0f,0.0f};
    const float NORMALIZATION[3] = {0.003921569f,0.003921569f,0.003921569f};
};

#endif /* Face_hpp */
