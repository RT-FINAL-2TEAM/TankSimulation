package com.tankcc.phonesim2real;

public class Detection {
    public String label;
    public float confidence;
    public float x1;
    public float y1;
    public float x2;
    public float y2;
    public Float distanceM;

    public Detection(String label, float confidence, float x1, float y1, float x2, float y2, Float distanceM) {
        this.label = label;
        this.confidence = confidence;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.distanceM = distanceM;
    }
}
