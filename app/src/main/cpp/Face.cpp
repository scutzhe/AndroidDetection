#include "Face.hpp"

Face::Face(std::string mnn_path){
    //model_configuration
    key_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    config.numThread = THREADS;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)0;
    config.backendConfig = &backendConfig;
    key_session = key_interpreter->createSession(config);

    //image_configuration
    image_config.sourceFormat = (MNN::CV::ImageFormat)0;//RGBA
    image_config.destFormat = (MNN::CV::ImageFormat)2;//BGR
    ::memcpy(image_config.mean, MEAN, sizeof(MEAN));
    ::memcpy(image_config.normal, NORMALIZATION, sizeof(NORMALIZATION));
}

float* Face:: detection(unsigned char *image_data, int width, int height, int channel) {
    MNN::Tensor* input_tensor = key_interpreter->getSessionInput(key_session, nullptr);
    MNN::CV::Matrix transform;
    std::vector<int>dims = { 1, CHANNELS, HEIGHT, WIDTH };
    key_interpreter->resizeTensor(input_tensor, dims);
    key_interpreter->resizeSession(key_session);
    transform.postScale(1.0f/(float)HEIGHT, 1.0f/(float)WIDTH);
    transform.postScale((float)width, (float)height);
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(
            image_config.sourceFormat, image_config.destFormat, image_config.mean,
            3, image_config.normal, 3));
    process->setMatrix(transform);
    process->convert(image_data, width, height, width*channel, input_tensor);
    key_interpreter->runSession(key_session);
    std::string output_tensor_name = "conv8_fwd";
    MNN::Tensor *tensor_landmarks  = key_interpreter->getSessionOutput(key_session, output_tensor_name.c_str());
    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);
    auto landmarks_data  = tensor_landmarks_host.host<float>();
    return landmarks_data;
}