#ifndef Face_hpp
#define Face_hpp
#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "mnn/Interpreter.hpp"
#include "mnn/MNNDefine.h"
#include "mnn/Tensor.hpp"
#include "mnn/ImageProcess.hpp"
#include "opencv2/opencv.hpp"
#define LOGInfo(...) __android_log_print(ANDROID_LOG_INFO,KEY_TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, KEY_TAG, __VA_ARGS__)

using namespace  std;

#define KEY_TAG "KEYPOINT"

class Face {
public:
    Face(std::string model_path);
    static cv::Mat transBufferToMat(unsigned char* pBuffer, int width, int height, int channel, int nBPB);
    float * detection(unsigned char *raw_image, int width, int height, int channel);

private:
    std::string model_path;
    std::shared_ptr<MNN::Interpreter> key_interpreter;
    MNN::Session *key_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    int WIDTH = 96;
    int HEIGHT = 96;
    int CHANNELS = 3;
    int THREADS = 4;
    const float MEAN[3] = {123.0,123.0,123.0};
    const float NORMALIZATION[3] = {1.0/58.0,1.0/58.0,1.0/58.0};
};

#endif /* Face_hpp */
