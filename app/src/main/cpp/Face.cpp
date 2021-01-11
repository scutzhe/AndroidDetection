#include "Face.hpp"

Face::Face(std::string mnn_path){
    //model_configuration
    face_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    config.numThread = THREADS;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)0;
    config.backendConfig = &backendConfig;
    face_session = face_interpreter->createSession(config);

    //image_configuration
    image_config.sourceFormat = (MNN::CV::ImageFormat)0;//RGBA
    image_config.destFormat = (MNN::CV::ImageFormat)2;// 1->RGB,2->BGR
    ::memcpy(image_config.mean, MEAN, sizeof(MEAN));
    ::memcpy(image_config.normal, NORMALIZATION, sizeof(NORMALIZATION));
}

float Face::IOU(FaceInfo boxes_one, FaceInfo boxes_two) {
    float x_min = std::max(boxes_one.x_min,boxes_two.x_min);
    float y_min = std::max(boxes_one.y_min,boxes_two.y_min);
    float x_max = std::min(boxes_one.x_max,boxes_two.x_max);
    float y_max = std::min(boxes_one.y_max,boxes_two.y_max);
    if(x_min > x_max || y_min > y_max)
        return 0.0f;
    else{
        float area_and = (x_max - x_min) * (y_max - y_min);
        float area_or = (boxes_one.x_max-boxes_one.x_min)*(boxes_one.y_max-boxes_one.y_min) +
                        (boxes_two.x_max-boxes_two.x_min)*(boxes_two.y_max-boxes_two.y_min) -
                        area_and;
        return area_and/area_or;
    }
}

bool Face::sort_score(FaceInfo boxes_one, FaceInfo boxes_two) {
    return boxes_one.score > boxes_two.score;
}

std::vector<FaceInfo> Face::NMS(std::vector<FaceInfo> boxes, float threshold) {
    std::sort(boxes.begin(),boxes.end(),sort_score);
    std::vector<FaceInfo>res;
    int N = boxes.size();
    std::vector<int> labels(N,-1);
    for(int i=0;i<N-1;++i){
        for(int j=i+1;j<N;++j){
            float iou = IOU(boxes[i],boxes[j]);
            if(iou > threshold){
                labels[j]=0;
            }
        }
    }
    for(int i=0;i<N;i++){
        if(labels[i] == -1){
            res.push_back(boxes[i]);
        }
    }
    return res;
}
float* Face:: detection(unsigned char *image_data, int width, int height, int channel) {
    MNN::Tensor* input_tensor = face_interpreter->getSessionInput(face_session, nullptr);
    MNN::CV::Matrix transform;
    std::vector<int>dims = { 1, CHANNELS, HEIGHT, WIDTH };
    face_interpreter->resizeTensor(input_tensor, dims);
    face_interpreter->resizeSession(face_session);
    transform.postScale(1.0f/(float)width, 1.0f/(float)height);
    transform.postScale((float)width, (float)height);
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(
            image_config.sourceFormat, image_config.destFormat, image_config.mean,
            3, image_config.normal, 3));
    process->setMatrix(transform);
    process->convert(image_data, width, height, width*channel, input_tensor);
    face_interpreter->runSession(face_session);
    std::string boxes = "Squeeze";
    std::string scores = "convert_scores";
    std::string anchors = "anchors";

    MNN::Tensor *tensor_boxes  = face_interpreter->getSessionOutput(face_session, boxes.c_str());
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    auto boxes_data  = tensor_boxes_host.host<float>();

    MNN::Tensor *tensor_scores  = face_interpreter->getSessionOutput(face_session, scores.c_str());
    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    auto scores_data  = tensor_scores_host.host<float>();

    MNN::Tensor *tensor_anchors  = face_interpreter->getSessionOutput(face_session,anchors.c_str());
    MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());
    tensor_anchors->copyToHostTensor(&tensor_anchors_host);
    auto anchors_data  = tensor_anchors_host.host<float>();

    std::vector<FaceInfo>boxes_tmp;
    for(int i=0;i<OUTPUT_NUM;++i){
        float score_background = exp(scores_data[i*2]);
        float score_foreground = exp(scores_data[i*2+1]);
        float score = score_foreground/(score_foreground + score_background);
        if(score > score_threshold){
            FaceInfo  faceInfo;
            float y_center =  boxes_data[i*4 + 0] / Y_SCALE  * anchors_data[i*4 + 2] + anchors_data[i*4 + 0];
            float x_center =  boxes_data[i*4 + 1] / X_SCALE  * anchors_data[i*4 + 3] + anchors_data[i*4 + 1];
            float h  = exp(boxes_data[i*4 + 2] / H_SCALE) * anchors_data[i*4 + 2];
            float w  = exp(boxes_data[i*4 + 3] / W_SCALE) * anchors_data[i*4 + 3];

            auto y_min  = ( y_center - h * 0.5 ) * height;
            auto x_min  = ( x_center - w * 0.5 ) * width;
            auto y_max  = ( y_center + h * 0.5 ) * height;
            auto x_max  = ( x_center + w * 0.5 ) * width;

            if(x_min <=0 || y_min <= 0 || x_max <= 0 || y_max <= 0){
                continue;;
            }
            faceInfo.x_min = x_min;
            faceInfo.y_min = y_min;
            faceInfo.x_max = x_max;
            faceInfo.y_max = y_max;
            faceInfo.score = score;
            boxes_tmp.push_back(faceInfo);
        }
    }
    std::vector<FaceInfo>res = NMS(boxes_tmp,nms_threshold);
    auto result = new float[5*res.size()];
    for(int i=0;i<res.size();i++){
        result[5*i] = res[i].x_min;
        result[5*i+1] = res[i].y_min;
        result[5*i+2] = res[i].x_max;
        result[5*i+3] = res[i].y_max;
        result[5*i+4] = res[i].score;
        LOGD("x_min=%.4f,y_min=%.4f,x_max=%.4f,y_max=%.4f,score=%.4f",
                res[i].x_min,res[i].y_min,res[i].x_max,res[i].y_max,res[i].score);
    }
    return result;
}