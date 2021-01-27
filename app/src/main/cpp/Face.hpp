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
    //人脸检测
    Face(const std::string face_model_path,const std::string key_model_path);
    static float IOU(FaceInfo boxes_one,FaceInfo boxes_two);
    static bool sort_score(FaceInfo boxes_one,FaceInfo boxes_two);
    static std::vector<FaceInfo> NMS(std::vector<FaceInfo> boxes,float threshold);
    float* face_detection(unsigned char *image_data, int width, int height, int channel);

    //关键点检测
    float* key_detection(unsigned char *image_data, int width, int height, int channel);

private:
    //人脸检测
    std::shared_ptr<MNN::Interpreter>face_interpreter = nullptr;
    MNN::Session *face_session = nullptr;
    MNN::CV::ImageProcess::Config face_image_config;
    MNN::ScheduleConfig face_config;
    MNN::BackendConfig face_backendConfig;

    int WIDTH = 128;
    int HEIGHT = 128;
    int CHANNELS = 3;
    int THREADS = 4;
    int OUTPUT_NUM = 960;
    float X_SCALE = 10.0;
    float Y_SCALE = 10.0;
    float H_SCALE = 5.0;
    float W_SCALE = 5.0;
    float score_threshold = 0.45f;
    float nms_threshold = 0.45f;
    const float MEAN[3] = {0.0f,0.0f,0.0f};
    const float NORMALIZATION[3] = {0.003921569f,0.003921569f,0.003921569f};

    //人脸关键点检测
    std::shared_ptr<MNN::Interpreter>key_interpreter = nullptr;
    MNN::Session *key_session = nullptr;
    MNN::CV::ImageProcess::Config key_image_config;
    MNN::ScheduleConfig key_config;
    MNN::BackendConfig key_backendConfig;

    int KEY_WIDTH = 96;
    int KEY_HEIGHT = 96;
    int KEY_CHANNELS = 3;
    int KEY_THREADS = 4;
    const float KEY_MEAN[3] = {0.0f,0.0f,0.0f};
    const float KEY_NORMALIZATION[3] = {0.003921569f,0.003921569f,0.003921569f};
};
#endif /* Face_hpp */