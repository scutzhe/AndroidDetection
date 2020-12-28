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
    string str_face = tFaceModelDir + "face_320.mnn";
    string str_keyPoint = tFaceModelDir + "keyPoint_96.mnn";
    face_detection = new  Face(str_face, str_keyPoint, 320, 240, 4, 0.65 ); // config model input
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
        LOGD("img data format error");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("img data is null");
        return NULL;
    }

    if(imageWidth<80||imageHeight<80){
        LOGD("img is too small");
        return NULL;
    }

    //detect face
    LOGD("imageWidth=%d, imageHeight=%d,imageChannel=%d",imageWidth,imageHeight,imageChannel);
    std::vector<FaceInfo> face_information = face_detection ->face_detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel);
    int32_t num_face = static_cast<int32_t>(face_information.size());
    int out_size = 1+num_face*4;
    int *face_box = new int[out_size];
    face_box[0] = num_face;
    for (int i=0; i<num_face; i++) {
        face_box[4*i+1] = (int)face_information[i].x1;
        face_box[4*i+2] = (int)face_information[i].y1;
        face_box[4*i+3] = (int)face_information[i].x2;
        face_box[4*i+4] = (int)face_information[i].y2;
    }

    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo, 0, out_size, face_box);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    delete [] face_box;
    return tFaceInfo;
}

JNIEXPORT jfloatArray JNICALL
Java_com_facesdk_FaceSDKNative_KeyPointDetection(JNIEnv *env, jobject instance, jbyteArray imageDate_,
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
        LOGD("img data format error");
        return NULL;
    }
    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("img data is null");
        return NULL;
    }
    if(imageWidth<24||imageHeight<24){
        LOGD("img is too small");
        return NULL;
    }

    //face keyPoint detection
    LOGD("imageWidth=%d, imageHeight=%d,imageChannel=%d",imageWidth,imageHeight,imageChannel);
    float* result = face_detection ->keyPoint_detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel);
    int out_size = 196;
    jfloatArray tFaceKeyPointInfo = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(tFaceKeyPointInfo, 0, out_size, result);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);

    delete [] result;
    return tFaceKeyPointInfo;
}

JNIEXPORT jboolean JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetectionModelUnInit(JNIEnv *env, jobject instance) {
    jboolean tDetectionUnInit = false;
    if (!detection_sdk_init_ok) {
        LOGD("sdk not inited, do nothing");
        return true;
    }
    delete face_detection;
    detection_sdk_init_ok = false;
    tDetectionUnInit = true;
    LOGD("sdk release ok");
    return tDetectionUnInit;
}
}
