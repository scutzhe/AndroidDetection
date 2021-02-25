package com.facesdk;

public class SelfDefine {
    public float x_min = 0.0f;
    public float y_min = 0.0f;
    public float x_max = 0.0f;
    public float y_max = 0.0f;
    public float score = 0.0f;
    public int label = 0;

    public SelfDefine(float x_min, float y_min, float x_max, float y_max, float score, int label){
        this.x_min = x_min;
        this.y_min = y_min;
        this.x_max = x_max;
        this.y_max = y_max;
        this.score = score;
        this.label = label;
    }
}
