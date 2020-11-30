#ifndef Face_hpp
#define Face_hpp
#pragma once

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <android/log.h>

#include <iostream>
#include <cmath>
#include <fstream>

#define LOG_TAG "DETECTION"
#define LOGInfo(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)


using namespace std;
using namespace tflite;

typedef struct FaceInfo{
    float x_min;
    float y_min;
    float x_max;
    float y_max;
    float score;
}FaceInfo;

class Face {
public:
    Face(const char* model_path,std::string label_path);
    static float exp_composite(float x);
    static cv::Mat transBufferToMat(unsigned char* pBuffer, int width, int height, int channel, int nBPB);
    void detection(unsigned char *raw_image, int width, int height, int channel,std::vector<FaceInfo>&face_info);
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::vector<std::string> labels;
    int WIDTH = 320;
    int HEIGHT = 240;
    int CHANNELS = 3;
    float SCORE_THRESHOLD= 0.75f;
};
#endif /* Face_hpp */
