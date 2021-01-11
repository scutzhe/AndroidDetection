#include "Face.hpp"

Face::Face(std::string mnn_path){
    //model_configuration
    key_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
//    config.type = static_cast<MNNForwardType>(MNN_FORWARD_VULKAN);
    config.numThread = THREADS;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)0;
    config.backendConfig = &backendConfig;
    key_session = key_interpreter->createSession(config);

    //image_configuration
    image_config.sourceFormat = (MNN::CV::ImageFormat)0;//RGBA
    image_config.destFormat = (MNN::CV::ImageFormat)2;// 1->RGB,2->BGR
    ::memcpy(image_config.mean, MEAN, sizeof(MEAN));
    ::memcpy(image_config.normal, NORMALIZATION, sizeof(NORMALIZATION));
}

//测试不带角度的人脸关键点
//float* Face:: detection(unsigned char *image_data, int width, int height, int channel) {
//    MNN::Tensor* input_tensor = key_interpreter->getSessionInput(key_session, nullptr);
//    MNN::CV::Matrix transform;
//    std::vector<int>dims = { 1, CHANNELS, HEIGHT, WIDTH };
//    key_interpreter->resizeTensor(input_tensor, dims);
//    key_interpreter->resizeSession(key_session);
//    transform.postScale(1.0f/(float)HEIGHT, 1.0f/(float)WIDTH);
//    transform.postScale((float)width, (float)height);
//    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(
//            image_config.sourceFormat, image_config.destFormat, image_config.mean,
//            3, image_config.normal, 3));
//    process->setMatrix(transform);
//    process->convert(image_data, width, height, width*channel, input_tensor);
//    key_interpreter->runSession(key_session);
//    std::string output_tensor_name = "conv8_fwd";
//    MNN::Tensor *tensor_landmarks  = key_interpreter->getSessionOutput(key_session, output_tensor_name.c_str());
//    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
//    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);
//    auto landmarks_data  = tensor_landmarks_host.host<float>();
//    return landmarks_data;
//}

////测试带角度的人脸关键点
float* Face:: detection(unsigned char *image_data, int width, int height, int channel) {
    MNN::Tensor* input_tensor = key_interpreter->getSessionInput(key_session, nullptr);
    MNN::CV::Matrix transform;
    std::vector<int>dims = { 1, CHANNELS, HEIGHT, WIDTH };
    key_interpreter->resizeTensor(input_tensor, dims);
    key_interpreter->resizeSession(key_session);
    transform.postScale(1.0f/(float)width, 1.0f/(float)height);
    transform.postScale((float)width, (float)height);
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(
            image_config.sourceFormat, image_config.destFormat, image_config.mean,
            3, image_config.normal, 3));
    process->setMatrix(transform);
    process->convert(image_data, width, height, width*channel, input_tensor);
    key_interpreter->runSession(key_session);
    std::string landmarks = "landmarks";
    std::string angle = "angle";
    float* result = new float[199]{0.0f};
    MNN::Tensor *tensor_landmarks  = key_interpreter->getSessionOutput(key_session, landmarks.c_str());
    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);
    auto landmarks_data  = tensor_landmarks_host.host<float>();
    MNN::Tensor *tensor_angle  = key_interpreter->getSessionOutput(key_session, angle.c_str());
    MNN::Tensor tensor_angle_host(tensor_landmarks, tensor_angle->getDimensionType());
    tensor_angle->copyToHostTensor(&tensor_angle_host);
    auto angle_data  = tensor_angle_host.host<float>();
    memcpy(result,landmarks_data,196 * sizeof(landmarks_data[0]));
    memcpy(&result[196],angle_data,3 * sizeof(angle_data[0]));
    return result;
}

////测试人脸
//float* Face:: detection(unsigned char *image_data, int width, int height, int channel) {
//    MNN::Tensor* input_tensor = key_interpreter->getSessionInput(key_session, nullptr);
//    MNN::CV::Matrix transform;
//    std::vector<int>dims = { 1, CHANNELS, HEIGHT, WIDTH };
//    key_interpreter->resizeTensor(input_tensor, dims);
//    key_interpreter->resizeSession(key_session);
//    transform.postScale(1.0f/(float)width, 1.0f/(float)height);
//    transform.postScale((float)width, (float)height);
//    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(
//            image_config.sourceFormat, image_config.destFormat, image_config.mean,
//            3, image_config.normal, 3));
//    process->setMatrix(transform);
//    process->convert(image_data, width, height, width*channel, input_tensor);
//    key_interpreter->runSession(key_session);
//    std::string locations = "TFLite_Detection_PostProcess";
//    std::string scores = "TFLite_Detection_PostProcess:2";
//    float* result = new float[50]{0.0f};
//    MNN::Tensor *tensor_locations  = key_interpreter->getSessionOutput(key_session, locations.c_str());
//    MNN::Tensor tensor_locations_host(tensor_locations, tensor_locations->getDimensionType());
//    tensor_locations->copyToHostTensor(&tensor_locations_host);
//    auto location_data  = tensor_locations_host.host<float>();
//
//    MNN::Tensor *tensor_scores  = key_interpreter->getSessionOutput(key_session, scores.c_str());
//    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
//    tensor_scores->copyToHostTensor(&tensor_scores_host);
//    auto score_data  = tensor_scores_host.host<float>();
//
//    memcpy(result,location_data,40 * sizeof(location_data[0]));
//    memcpy(&result[40],score_data,10 * sizeof(score_data[0]));
//    return result;
//}