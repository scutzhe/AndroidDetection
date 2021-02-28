#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include "Face.hpp"

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
    string model_path = tFaceModelDir + "face.mnn";
//    string model_path = tFaceModelDir + "face_quant.mnn";
    face_detection = new  Face(model_path);
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

    //detect face
    LOGD("imageWidth=%d, imageHeight=%d,imageChannel=%d",imageWidth,imageHeight,imageChannel);
    float* result = face_detection ->detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel);
    int out_size = int(result[0]);
    jfloatArray tFaceInfo = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(tFaceInfo, 0, out_size, result);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
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
