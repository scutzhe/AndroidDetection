package com.facesdk;

public class FaceSDKNative {
    //SDK初始化
    public native boolean FaceDetectionModelInit(String faceDetectionModelPath);

    //SDK人脸关键点检测接口
    public native float[] FaceDetection(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);

    //SDK销毁
    public native boolean FaceDetectionModelUnInit();

    static {
        System.loadLibrary("face_detection");
    }

}
