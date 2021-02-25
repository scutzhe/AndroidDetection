#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include "litedet.hpp"


#define TAG "FaceSDKNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

using namespace std;
static LiteDet *lite_detection;
bool detection_sdk_init_ok = false;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetectionModelInit(JNIEnv *env, jobject instance,
                                                      jstring faceDetectionModelPath_) {
    LOGD("JNI init native sdk");
    if (detection_sdk_init_ok) {
        LOGD("sdk already init");
        return true;
    }
    jboolean tRet = false;
    if (nullptr == faceDetectionModelPath_) {
        LOGD("model dir is empty");
        return tRet;
    }

    //get model absolute path
    const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
    if (nullptr == faceDetectionModelPath) {
        LOGD("model dir is empty");
        return tRet;
    }

    string tFaceModelDir = faceDetectionModelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
    string model_path = tFaceModelDir + "face_lite.mnn";
    lite_detection = new LiteDet(model_path);
    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
    tRet = true;
    return tRet;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetection(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                            jint imageWidth, jint imageHeight, jint imageChannel) {
    if(!detection_sdk_init_ok){
        LOGD("sdk not init");
        return nullptr;
    }
    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("imgW=%d, imgH=%d,imgC=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("img data format error");
        return nullptr;
    }
    jbyte *imageDate = env->GetByteArrayElements(imageDate_, nullptr);
    if (nullptr == imageDate){
        LOGD("img data is nullptr");
        return nullptr;
    }
    if(imageWidth<24||imageHeight<24){
        LOGD("img is too small");
        return nullptr;
    }

    //detect face
    LOGD("imageWidth=%d, imageHeight=%d,imageChannel=%d",imageWidth,imageHeight,imageChannel);
    std::vector<float>result = lite_detection ->detection((unsigned char*)imageDate, imageWidth, imageHeight, imageChannel);
    int out_size = result.size() ;
    float* jni_result = new float[out_size];
    if(!result.empty()){
        memcpy(jni_result,&result[0],result.size()*sizeof(float));
    }
    jfloatArray FaceInfo = env->NewFloatArray(out_size);
    env->SetFloatArrayRegion(FaceInfo, 0, out_size, jni_result);
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return FaceInfo;
}

// 返回数组, 其类型是结构体
//extern "C" JNIEXPORT jobjectArray JNICALL
//Java_com_facesdk_FaceSDKNative_FaceDetection(JNIEnv *env, jobject instance, jbyteArray imageDate_,
//                                             jint imageWidth, jint imageHeight, jint imageChannel) {
//    if (!detection_sdk_init_ok) {
//        LOGD("sdk not init");
//        return nullptr;
//    }
//
//    int tImageDateLen = env->GetArrayLength(imageDate_);
//    if (imageChannel == tImageDateLen / imageWidth / imageHeight) {
//        LOGD("imgW=%d, imgH=%d,imgC=%d", imageWidth, imageHeight, imageChannel);
//    } else {
//        LOGD("img data format error");
//        return nullptr;
//    }
//
//    jbyte *imageDate = env->GetByteArrayElements(imageDate_, nullptr);
//    if (nullptr == imageDate) {
//        LOGD("img data is nullptr");
//        return nullptr;
//    }
//
//    if (imageWidth < 96 || imageHeight < 96) {
//        LOGD("img is too small");
//        return nullptr;
//    }
//
//    //face_detection
//    LOGD("imageWidth=%d,imageHeight=%d,imageChannel=%d", imageWidth, imageHeight, imageChannel);
//    std::vector<BoxInfo> result = lite_detection->detection((unsigned char *) imageDate, imageWidth,
//                                                            imageHeight, imageChannel);
//    int num = result.size();
//    LOGD("jni_num=%d", num);
//
//    jobject _obj;
//    // 获取object所属类,一般初始值为java/lang/Object
//    jclass objClass = (env)->FindClass("java/lang/Object");
//    // 新建object数组对象
//    jobjectArray jniResult = (env)->NewObjectArray(num, objClass, NULL);
//    // 获取java中的实例类
//    jclass objectClass = (env)->FindClass("com/facesdk/SelfDefine");
//    // 获取类中每个变量的定义
//    jfieldID x_min = (env)->GetFieldID(objectClass, "x_min", "F");
//    jfieldID y_min = (env)->GetFieldID(objectClass, "y_min", "F");
//    jfieldID x_max = (env)->GetFieldID(objectClass, "x_max", "F");
//    jfieldID y_max = (env)->GetFieldID(objectClass, "y_max", "F");
//    jfieldID score = (env)->GetFieldID(objectClass, "score", "F");
//    jfieldID label = (env)->GetFieldID(objectClass, "label", "I");
//    for (int i = 0; i < num; i++) {
//        // 给每一个实例变量赋值
//        (env)->SetFloatField(_obj,x_min,result[i].x1);
//        (env)->SetFloatField(_obj,y_min,result[i].y1);
//        (env)->SetFloatField(_obj,x_max,result[i].x2);
//        (env)->SetFloatField(_obj,y_max,result[i].y2);
//        (env)->SetFloatField(_obj,score,result[i].score);
//        (env)->SetIntField(_obj,label,result[i].label);
//        // 添加到object数组中
//        (env)->SetObjectArrayElement(jniResult,i,_obj);
//    }
//    return jniResult;
//}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_facesdk_FaceSDKNative_FaceDetectionModelUnInit(JNIEnv *env, jobject instance) {
    jboolean tDetectionUnInit = false;
    if (!detection_sdk_init_ok) {
        LOGD("sdk not init, do nothing");
        return true;
    }
    delete lite_detection;
    detection_sdk_init_ok = false;
    tDetectionUnInit = true;
    LOGD("sdk release ok");
    return tDetectionUnInit;
}
