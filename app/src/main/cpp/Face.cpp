# include "Face.hpp"

Face::Face(const char* model_path,std::string label_path){
    model = tflite::FlatBufferModel::BuildFromFile(model_path);
    std::ifstream input(label_path);
    for( std::string line; getline( input, line ); )
    {
        labels.push_back( line);
    }
}

float Face:: exp_composite(float x) {
    return 1.f / (1.f + expf(-x));
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

void Face::detection(unsigned char *raw_image, int width, int height, int channel,std::vector<FaceInfo>&face_info) {
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    // Resize input tensors, if desired.
    TfLiteTensor* output_locations = nullptr;
    TfLiteTensor* output_classes = nullptr;
    TfLiteTensor* output_scores = nullptr;
    TfLiteTensor* num_detections = nullptr;

    //input image
    cv::Mat image_input;
    cv::Mat image = transBufferToMat(raw_image,width,height,channel,1);
    cv::resize(image,image_input,cv::Size(WIDTH,HEIGHT));
    cv::cvtColor(image_input,image_input,CV_BGR2RGB);
    interpreter->AllocateTensors();

    //get input
    auto *input_tensor = interpreter->typed_input_tensor<float>(0);
    for(int i=0;i<image_input.cols * image_input.rows * CHANNELS;i++){
        input_tensor[i] = image_input.data[i]/255.0;
    }
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(6);

    // execute inference
    interpreter->Invoke();

    //get result
    output_locations = interpreter->tensor(interpreter->outputs()[0]);
    output_classes  = interpreter->tensor(interpreter->outputs()[1]);
    output_scores = interpreter->tensor(interpreter->outputs()[2]);
    num_detections   = interpreter->tensor(interpreter->outputs()[3]);

    auto out_loc = output_locations->data.f;
    auto out_cls = output_classes->data.f;
    auto out_scores = output_scores->data.f;
    auto out_num = num_detections->data.f;

    std::vector<float> locations;
    std::vector<float> cls;
    std::vector<float> score;

    for (int i = 0; i < 10; i++){
        locations.push_back(out_loc[i]);
        cls.push_back(out_cls[i]);
        score.push_back(out_scores[i]);
    }

    int count=0;
    for(std::size_t j = 0; j <locations.size(); j+=4){
        float score = exp_composite(out_num[count]);
        if (score < SCORE_THRESHOLD){
            continue;
        }
        float y_min=locations[j]* float(image.rows);
        float x_min=locations[j+1]* float(image.cols);
        float y_max=locations[j+2]* float(image.rows);
        float x_max=locations[j+3]* float(image.cols);
        if(y_min < 0 || x_min < 0 || y_max < 0 || x_max < 0){
            continue;
        }
        FaceInfo  box_score;
        box_score.x_min = x_min;
        box_score.y_min = y_min;
        box_score.x_max = x_max;
        box_score.y_max = y_max;
        box_score.score = score;
        face_info.push_back(box_score);
        count+=1;
    }
}