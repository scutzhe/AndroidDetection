#ifndef __LiteDet_H__
#define __LiteDet_H__
#pragma once

#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <jni.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, FACE_TAG, __VA_ARGS__)
#define FACE_TAG "LITE"


typedef struct HeadInfo_
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo_
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class LiteDet {
public:
    LiteDet(const std::string &mnn_path);
    ~LiteDet();
     std::vector<float> detection(unsigned char *image_data, int width, int height, int channel);

private:
    void decode_infer(MNN::Tensor *cls_pred, MNN::Tensor *dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>> &results);
    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);
    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

    std::shared_ptr<MNN::Interpreter> liteDet_interpreter=nullptr;
    MNN::Session *liteDet_session = nullptr;
    MNN::CV::ImageProcess::Config image_config;
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;
    MNN::Tensor *input_tensor = nullptr;

    const int WIDTH = 128;
    const int HEIGHT = 128;
    const int CHANNELS = 3;
    const int THREADS = 4;
    const float SCORE_THRESHOLD = 0.3;
    const float NMS_THRESHOLD = 0.3;
    const float MEAN[3] = { 103.53f, 116.28f, 123.675f };
    const float NORMALIZATION[3] = { 0.017429f, 0.017507f, 0.017125f };
    const int NUM_CLASSES = 2;
    const int REG_MAX = 7;

    std::vector<HeadInfo> heads_info{
            {"1023", "1026", 8},
            {"1045", "1048", 16},
            {"1067", "1070", 32},
    };
    std::vector<std::string>labels{"face", "hand"};
};

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);
inline float fast_exp(float x);
#endif