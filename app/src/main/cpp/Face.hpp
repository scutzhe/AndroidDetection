#ifndef Face_hpp
#define Face_hpp
#pragma once

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <ctime>
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include "Matrix.h"
#include "net.h"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2
typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

class Face {
public:
    Face(std::string &detection_mnn_path, std::string &keyPoint_mnn_path, int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.35);
    std::vector<FaceInfo> face_detection(unsigned char *raw_image, int width, int height, int channel);
    float * keyPoint_detection(unsigned char *image_data, int width, int height, int channel);
    void transform_buffer(unsigned char *data, unsigned char* data_crop, int x_min, int y_min, int crop_width, int crop_height, int width, int channel);
    void generateBBox(std::vector<FaceInfo> &bbox_collection,  float* scores, float* boxes);
    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);
    void Delay(int  time);
private:
    //face detection
    Inference_engine Face_net;
    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    float score_threshold;
    float iou_threshold;

    float mean_vals[3] = {127.0f, 127.0f, 127.0f};
    float norm_vals[3] = {1.0 / 128.0f, 1.0 / 128.0f, 1.0 / 128.0f};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
                        {10.0f,  16.0f,  24.0f},
                        {32.0f,  48.0f},
                        {64.0f,  96.0f},
                        {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;
    std::vector<std::vector<float>> priors = {};

    // keyPoint detection
    std::shared_ptr<MNN::Interpreter>key_interpreter = nullptr;
    MNN::Session *key_session = nullptr;
    MNN::CV::ImageProcess::Config image_config;
    MNN::ScheduleConfig config;
    MNN::BackendConfig backendConfig;

    int WIDTH = 96;
    int HEIGHT = 96;
    int CHANNELS = 3;
    int THREADS = 4;
    const float MEAN[3] = {123.0f,123.0f,123.0f};
    const float NORMALIZATION[3] = {1.0 / 58.0f,1.0 / 58.0f , 1.0 / 58.0f};
};
#endif /*Face_hpp*/