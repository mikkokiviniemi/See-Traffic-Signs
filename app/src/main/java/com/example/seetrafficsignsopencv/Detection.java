package com.example.seetrafficsignsopencv;

import org.opencv.core.Point;

public class Detection {
    ImageClass imgClass;
    Point startPoint;
    Point endPoint;
    Float confidence;

    public Detection(ImageClass imgClass, Point startPoint, Point endPoint, Float confidence) {
        this.imgClass = imgClass;
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.confidence = confidence;
    }

}
