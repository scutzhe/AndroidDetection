#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include "Face.hpp"

#define TAG "FaceSDKNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

using namespace std;

static Face *face_detection;
bool detection_sdk_init_ok = false;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetectionModelInit(JNIEnv *env, jobject instance,
                                                      jstring faceDetectionModelPath_) {
    LOGD("JNI init native sdk");
    if (detection_sdk_init_ok) {
        LOGD("sdk already init");
        return true;
    }
    jboolean tRet = false;
    if (NULL == faceDetectionModelPath_) {
        LOGD("model dir is empty");
        return tRet;
    }

    //get model absolute path
    const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
    if (NULL == faceDetectionModelPath) {
        LOGD("model dir is empty");
        return tRet;
    }

    string tFaceModelDir = faceDetectionModelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length()-1, 1);
    string model_path = tFaceModelDir + "face.tflite";
    const char* const_model_path = model_path.c_str();
    string label_path = tFaceModelDir + "label_map_face.txt";
    face_detection = new Face(const_model_path,label_path); // config model input
    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
    tRet = true;
    return tRet;
}

JNIEXPORT jintArray JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetection(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                          jint imageWidth, jint imageHeight, jint imageChannel) {
    if(!detection_sdk_init_ok){
        LOGD("sdk not init");
        return NULL;
    }

    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("imgW=%d, imgH=%d,imgC=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("image data format error");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("image data is null");
        return NULL;
    }

    if(imageWidth<160||imageHeight<120){
        LOGD("image is too small");
        return NULL;
    }

    std::vector<FaceInfo> face_info;
    //get result
    face_detection ->detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel,face_info);
    int32_t detection_num = static_cast<int32_t>(face_info.size());
    int out_size = 1+detection_num*4;
    int *faceInfo = new int[out_size];
    faceInfo[0] = detection_num;

    for (int i=0; i<detection_num; i++) {
        faceInfo[4*i+1] = int(face_info[i].x_min);
        faceInfo[4*i+2] = int(face_info[i].y_min);
        faceInfo[4*i+3] = int(face_info[i].x_max);
        faceInfo[4*i+4] = int(face_info[i].y_max);
    }
    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo, 0, out_size, faceInfo);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);

    delete [] faceInfo;
    return tFaceInfo;
}

JNIEXPORT jboolean JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetectionModelUnInit(JNIEnv *env, jobject instance) {

    jboolean tDetectionUnInit = false;

    if (!detection_sdk_init_ok) {
        LOGD("sdk not init, do nothing");
        return true;
    }

    delete face_detection;

    detection_sdk_init_ok = false;

    tDetectionUnInit = true;

    LOGD("sdk release ok");

    return tDetectionUnInit;
}

}
