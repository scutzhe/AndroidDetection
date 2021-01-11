#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include "Face.hpp"

#define TAG "FaceSDKNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

using namespace std;
static Face *face;
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
    string str_face = tFaceModelDir + "face.mnn";
    string str_keyPoint = tFaceModelDir + "key_cpu.mnn";
    face = new  Face(str_face, str_keyPoint);
    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
    tRet = true;
    return tRet;
}

JNIEXPORT jfloatArray JNICALL
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

    if(imageWidth<96||imageHeight<96){
        LOGD("img is too small");
        return NULL;
    }

    //face_detection
    LOGD("imageWidth=%d, imageHeight=%d,imageChannel=%d",imageWidth,imageHeight,imageChannel);
    float* result = face ->face_detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel);
    int out_size = 5 * sizeof(result)/sizeof(result[0]);
    jfloatArray tFaceInfo = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(tFaceInfo, 0, out_size, result);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_facesdk_FaceSDKNative_KeyDetection(JNIEnv *env, jobject instance, jbyteArray imageDate_,
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

    //detect face
    LOGD("imageWidth=%d, imageHeight=%d,imageChannel=%d",imageWidth,imageHeight,imageChannel);
    float* result = face ->key_detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel);
    int out_size = 199;
    jfloatArray tFaceInfo = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(tFaceInfo, 0, out_size, result);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetectionModelUnInit(JNIEnv *env, jobject instance) {
    jboolean tDetectionUnInit = false;
    if (!detection_sdk_init_ok) {
        LOGD("sdk not inited, do nothing");
        return true;
    }
    delete face;
    detection_sdk_init_ok = false;
    tDetectionUnInit = true;
    LOGD("sdk release ok");
    return tDetectionUnInit;
}
