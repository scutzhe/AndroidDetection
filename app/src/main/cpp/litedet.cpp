#include "litedet.hpp"

LiteDet::LiteDet(const std::string &mnn_path)
{
    // MNN configuration
    liteDet_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    config.numThread = THREADS;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)0;
    config.backendConfig = &backendConfig;
    liteDet_session = liteDet_interpreter->createSession(config);
    input_tensor = liteDet_interpreter->getSessionInput(liteDet_session, nullptr);

    //image_configuration
    image_config.sourceFormat = (MNN::CV::ImageFormat)0;//RGBA
    image_config.destFormat = (MNN::CV::ImageFormat)2;//1->RGB,2->BGR
    ::memcpy(image_config.mean, MEAN, sizeof(MEAN));
    ::memcpy(image_config.normal, NORMALIZATION, sizeof(NORMALIZATION));
}

LiteDet::~LiteDet()
{
    liteDet_interpreter->releaseModel();
    liteDet_interpreter->releaseSession(liteDet_session);
}

std::vector<float> LiteDet::detection(unsigned char *image_data, int width, int height, int channel)
{
    MNN::CV::Matrix transform;
    std::vector<int>dims = { 1, CHANNELS, HEIGHT, WIDTH };
    liteDet_interpreter->resizeTensor(input_tensor, dims);
    liteDet_interpreter->resizeSession(liteDet_session);

    // image process
    transform.postScale(1.0f/(float)WIDTH, 1.0f/(float)HEIGHT);
    transform.postScale((float)width, (float)height);
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(
            image_config.sourceFormat, image_config.destFormat, image_config.mean,
            3, image_config.normal, 3));
    process->setMatrix(transform);
    process->convert(image_data, width, height, width*channel, input_tensor);

    // run network
    liteDet_interpreter->runSession(liteDet_session);

    // get middle result
    std::vector<std::vector<BoxInfo>> results_middle;
    results_middle.resize(NUM_CLASSES);

    // get last result
    std::vector<float> results;

    for (const auto &head_info : heads_info)
    {
        MNN::Tensor *tensor_scores = liteDet_interpreter->getSessionOutput(liteDet_session, head_info.cls_layer.c_str());
        MNN::Tensor *tensor_boxes = liteDet_interpreter->getSessionOutput(liteDet_session, head_info.dis_layer.c_str());

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
        tensor_scores->copyToHostTensor(&tensor_scores_host);

        MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
        tensor_boxes->copyToHostTensor(&tensor_boxes_host);

        decode_infer(tensor_scores, tensor_boxes, head_info.stride, SCORE_THRESHOLD, results_middle);
    }
    for (int i = 0; i < (int)results_middle.size(); i++) {
        nms(results_middle[i], NMS_THRESHOLD);
        for (auto box : results_middle[i]) {
            box.x1 = box.x1 / WIDTH * width;
            box.y1 = box.y1 / HEIGHT * height;
            box.x2 = box.x2 / WIDTH * width;
            box.y2 = box.y2 / HEIGHT * height;
            results.push_back(box.x1);
            results.push_back(box.y1);
            results.push_back(box.x2);
            results.push_back(box.y2);
            results.push_back(box.score);
            results.push_back((float)box.label);
        }
    }
    return results;
}

void LiteDet::decode_infer(MNN::Tensor *cls_pred, MNN::Tensor *dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>> &results)
{
    int feature_h = HEIGHT / stride;
    int feature_w = WIDTH / stride;

    for (int idx = 0; idx < feature_h * feature_w; idx++)
    {
        // scores is a tensor with shape [feature_h * feature_w, num_class]
        const float *scores = cls_pred->host<float>() + (idx * NUM_CLASSES);

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < NUM_CLASSES; label++)
        {
            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold)
        {
            // bbox is a tensor with shape [feature_h * feature_w, 4_points * 8_distribution_bite]
            const float *bbox_pred = dis_pred->host<float>() + (idx * 4 * (REG_MAX + 1));
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }
    }
}


BoxInfo LiteDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float *dis_after_sm = new float[REG_MAX + 1];
        activation_function_softmax(dfl_det + i * (REG_MAX + 1), dis_after_sm, REG_MAX + 1);
        for (int j = 0; j < REG_MAX + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)WIDTH);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)HEIGHT);

    return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void LiteDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

inline float fast_exp(float x)
{
    union{
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }
    return 0;
}
