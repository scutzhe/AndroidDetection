#include "Face.hpp"

Face::Face(std::string mnn_path){
    mnn_path = model_path;
    key_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = THREADS;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
    key_session = key_interpreter->createSession(config);
    input_tensor = key_interpreter->getSessionInput(key_session, nullptr);
}

cv::Mat Face:: transBufferToMat(unsigned char* pBuffer, int width, int height, int channel, int nBPB){
    cv::Mat mDst;
    if (channel == 3){
        if (nBPB == 1){
            mDst = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
        }
        else if (nBPB == 2){
            mDst = cv::Mat::zeros(cv::Size(width, height), CV_16UC3);
        }
    }
    else if (channel == 1){
        if (nBPB == 1){
            mDst = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
        }
        else if (nBPB == 2){
            mDst = cv::Mat::zeros(cv::Size(width, height), CV_16UC1);
        }
    }

    else if(channel == 4){
        if (nBPB == 1){
            mDst = cv::Mat::zeros(cv::Size(width, height), CV_8UC4);
        }
        else if (nBPB == 2){
            mDst = cv::Mat::zeros(cv::Size(width, height), CV_16UC4);
        }
    }

    for (int j = 0; j < height; ++j){
        unsigned char* data = mDst.ptr<unsigned char>(j);
        unsigned char* pSubBuffer = pBuffer + (height - 1 - j) * width  * channel * nBPB;
        memcpy(data, pSubBuffer, width * channel * nBPB);
    }
    if (channel == 3){
        cv::cvtColor(mDst, mDst, CV_RGB2BGR);
    }
    else if (channel == 1){
        cv::cvtColor(mDst, mDst, CV_GRAY2BGR);
    }
    else if (channel == 4){
        cv::cvtColor(mDst, mDst, CV_RGBA2BGR);
    }
    return mDst;
}

float* Face:: detection(unsigned char *raw_image, int width, int height, int channel) {
    //input_data
    cv::Mat image_input;
    cv::Mat image = transBufferToMat(raw_image,width,height,channel,1);
//    LOGInfo("image.cols=%d,image.rows=%d",image.cols,image.rows);
    cv::resize(image,image_input,cv::Size(WIDTH,HEIGHT),cv::INTER_CUBIC);

    // load and config mnn model
    key_interpreter->resizeTensor(input_tensor, {1, 3, HEIGHT, WIDTH});
    key_interpreter->resizeSession(key_session);
    std::shared_ptr<MNN::CV::ImageProcess> pre_treat(
            MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, MEAN, 3,
                                          NORMALIZATION, 3));
    pre_treat->convert(image.data, WIDTH, HEIGHT, image.step[0], input_tensor);

    // run inference
    key_interpreter->runSession(key_session);

    // get output data and no post deal
    std::string output_tensor_name = "conv5_fwd";
    MNN::Tensor *tensor_landmarks  = key_interpreter->getSessionOutput(key_session, output_tensor_name.c_str());
    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);
    auto landmarks_data  = tensor_landmarks_host.host<float>();
    return landmarks_data;
}