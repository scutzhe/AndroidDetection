#ifndef Face_hpp
#define Face_hpp
#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "Backend.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "revertMNNModel.hpp"
using namespace  std;

#define TAG "KEYPOINT"
#define LOGInfo(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

class Face {
public:
    Face(std::string model_path);
    static cv::Mat transBufferToMat(unsigned char* pBuffer, int width, int height, int channel, int nBPB);
    float * detection(unsigned char *raw_image, int width, int height, int channel);
private:
    std::string model_path;
    int WIDTH = 96;
    int HEIGHT = 96;
    int CHANNELS = 3;
    int THREADS = 1;
};

#endif /* Face_hpp */
