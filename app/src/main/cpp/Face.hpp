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

class Face {
public:
    Face(std::string model_path);
    float* detection(unsigned char *image_data, int width, int height, int channel);

private:
    std::shared_ptr<MNN::Interpreter>key_interpreter = nullptr;
    MNN::Session *key_session = nullptr;
    MNN::CV::ImageProcess::Config image_config;
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;

    int WIDTH = 96;
    int HEIGHT = 96;
    int CHANNELS = 3;
    int THREADS = 2;
//    const float MEAN[3] = {123.0f,123.0f,123.0f};
//    const float NORMALIZATION[3] = {58.0f,58.0f,58.0f};
    const float MEAN[3] = {0.0f,0.0f,0.0f};
    const float NORMALIZATION[3] = {0.003921569f,0.003921569f,0.003921569f};


};

#endif /* Face_hpp */
