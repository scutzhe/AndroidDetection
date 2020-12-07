#include "Face.hpp"

Face::Face(std::string mnn_path){
    mnn_path = model_path;
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
    LOGInfo("image.cols=%d,image.rows=%d",image.cols,image.rows);
    cv::resize(image,image_input,cv::Size(WIDTH,HEIGHT),cv::INTER_CUBIC);
    cv::cvtColor(image_input,image_input,CV_BGR2RGB);

    // load and config mnn model
    auto revertor = std::unique_ptr<Revert>(new Revert(model_path.c_str()));
    revertor->initialize();
    auto modelBuffer      = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    MNN::ScheduleConfig config;
    config.numThread = THREADS;
    config.type      = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    config.backendConfig = &backendConfig;

    auto session = net->createSession(config);
    net->releaseModel();

    clock_t start = clock();
    // preprocessing
    image.convertTo(image, CV_32FC3);
    image = (image - 123.0) / 58.0;

    // wrapping input tensor, convert nhwc to nchw
    std::vector<int> dims{1, HEIGHT, WIDTH, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    ::memcpy(nhwc_data, image.data, nhwc_size);

    std::string input_tensor = "data";
    auto inputTensor  = net->getSessionInput(session, nullptr);
    inputTensor->copyFromHostTensor(nhwc_Tensor);

    // run network
    net->runSession(session);

    // get output data
    std::string output_tensor_name0 = "conv5_fwd";
    MNN::Tensor *tensor_lmks  = net->getSessionOutput(session, output_tensor_name0.c_str());
    MNN::Tensor tensor_lmks_host(tensor_lmks, tensor_lmks->getDimensionType());
    tensor_lmks->copyToHostTensor(&tensor_lmks_host);

    // post processing steps
    auto lmks_dataPtr  = tensor_lmks_host.host<float>();
}