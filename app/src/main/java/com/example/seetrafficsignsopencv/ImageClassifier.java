package com.example.seetrafficsignsopencv;
/*
This software project uses MIT -license and uses 3rd party parts that are licensed via Apache License 2.0.
--------
Copyright 2022 Solita Oy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
import android.content.Context;
import android.graphics.Bitmap;

import com.example.seetrafficsignsopencv.ml.BWSmall;
import com.example.seetrafficsignsopencv.ml.BWLarge;
import com.example.seetrafficsignsopencv.ml.ColorSmall;
import com.example.seetrafficsignsopencv.ml.ColorLarge;

import com.example.seetrafficsignsopencv.ml.Detect;
import com.example.seetrafficsignsopencv.ml.Icmodel;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class ImageClassifier {

    private final Context context;

    private final int imageSizeLarge;
    private final int imageSizeSmall;

    private float detectionThreshold;

    public ImageClassifier(Context context) {

        this.imageSizeLarge = 640;
        this.imageSizeSmall = 320;

        this.detectionThreshold = (float) 0.20;
        this.context = context;
    }

    private ByteBuffer getByteBuffer(Bitmap bitmap) {
        int imgSize = bitmap.getWidth();
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[imgSize * imgSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < imgSize; i++) {
            for (int j = 0; j < imgSize; j++) {
                int val = intValues[pixel++]; // RGB
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }


    public Detection classifyImage(Mat image) {
        return this.classifyImageColorSmall(image);
    }

    public Detection classifyImageColorSmallOld(Mat image) {

        System.out.println("Detection with Old 1.1 Color 320x320");

        ImageClass imageClass = ImageClass.EMPTY;

        float dStartPointX = 0;
        float dStartPointY = 0;
        float dEndPointX = 0;
        float dEndPointY = 0;

        float bestConfidence = 0;

        try {
            Detect model = Detect.newInstance(context);

            // Konvertoidaan Mat -> Bitmap

            Bitmap origBitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image, origBitmap);
            Bitmap bitmap = Bitmap.createScaledBitmap(origBitmap, imageSizeSmall, imageSizeSmall, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSizeSmall, imageSizeSmall, 3}, DataType.FLOAT32);

            // Työnnetään bitmap inputfeatureen
            inputFeature0.loadBuffer(getByteBuffer(bitmap));

            // Runs model inference and gets result.
            Detect.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            float[] detectionPoints = outputFeature1.getFloatArray();
            int[] detections = outputFeature3.getIntArray();

            int bestIndex = detections[0];
            bestConfidence = confidences[0];

            imageClass = ImageClass.values()[bestIndex];

            dStartPointY = origBitmap.getHeight() * detectionPoints[0];
            dStartPointX = origBitmap.getWidth() * detectionPoints[1];
            dEndPointY = origBitmap.getHeight() * detectionPoints[2];
            dEndPointX = origBitmap.getWidth() * detectionPoints[3];

            if (bestConfidence < this.detectionThreshold) {
                System.out.println("DETECTION DEBUG: Confidence under threshold. Setting output to EMPTY");
                imageClass = ImageClass.EMPTY;
            }

            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return new Detection(imageClass, new Point(dStartPointX, dStartPointY), new Point(dEndPointX, dEndPointY), bestConfidence);

    }

    public Detection classifyImageColorSmall(Mat image) {

        System.out.println("Detection with Color 320x320");

        ImageClass imageClass = ImageClass.EMPTY;

        float dStartPointX = 0;
        float dStartPointY = 0;
        float dEndPointX = 0;
        float dEndPointY = 0;

        float bestConfidence = 0;

        try {
            ColorSmall model = ColorSmall.newInstance(context);

            // Konvertoidaan Mat -> Bitmap

            Bitmap origBitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image, origBitmap);
            Bitmap bitmap = Bitmap.createScaledBitmap(origBitmap, imageSizeSmall, imageSizeSmall, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSizeSmall, imageSizeSmall, 3}, DataType.FLOAT32);

            // Työnnetään bitmap inputfeatureen
            inputFeature0.loadBuffer(getByteBuffer(bitmap));

            // Runs model inference and gets result.
            ColorSmall.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            float[] detectionPoints = outputFeature1.getFloatArray();
            int[] detections = outputFeature3.getIntArray();

            int bestIndex = detections[0];
            bestConfidence = confidences[0];

            imageClass = ImageClass.values()[bestIndex];

            dStartPointY = origBitmap.getHeight() * detectionPoints[0];
            dStartPointX = origBitmap.getWidth() * detectionPoints[1];
            dEndPointY = origBitmap.getHeight() * detectionPoints[2];
            dEndPointX = origBitmap.getWidth() * detectionPoints[3];

            if (bestConfidence < this.detectionThreshold) {
                System.out.println("DETECTION DEBUG: Confidence under threshold. Setting output to EMPTY");
                imageClass = ImageClass.EMPTY;
            }

            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return new Detection(imageClass, new Point(dStartPointX, dStartPointY), new Point(dEndPointX, dEndPointY), bestConfidence);

    }

    public Detection classifyImageBWSmall(Mat image) {


        Mat image_bw = new Mat();
        Imgproc.cvtColor(image, image_bw, Imgproc.COLOR_BGR2GRAY);

        System.out.println("Detection with Grayscale 640x640");

        ImageClass imageClass = ImageClass.EMPTY;

        float dStartPointX = 0;
        float dStartPointY = 0;
        float dEndPointX = 0;
        float dEndPointY = 0;

        float bestConfidence = 0;

        try {
            BWSmall model = BWSmall.newInstance(context);

            // Konvertoidaan Mat -> Bitmap

            Bitmap origBitmap = Bitmap.createBitmap(image_bw.cols(), image_bw.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image_bw, origBitmap);
            Bitmap bitmap = Bitmap.createScaledBitmap(origBitmap, imageSizeSmall, imageSizeSmall, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSizeSmall, imageSizeSmall, 3}, DataType.FLOAT32);

            // Työnnetään bitmap inputfeatureen
            inputFeature0.loadBuffer(getByteBuffer(bitmap));

            // Runs model inference and gets result.
            BWSmall.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            float[] detectionPoints = outputFeature1.getFloatArray();
            int[] detections = outputFeature3.getIntArray();

            System.out.println("Confidences " + Arrays.toString(confidences));
            System.out.println("Detections " + Arrays.toString(detections));
            System.out.println("Detection points " + Arrays.toString(detectionPoints));

            int bestIndex = detections[0];
            bestConfidence = confidences[0];

            imageClass = ImageClass.values()[bestIndex];

            dStartPointY = origBitmap.getHeight() * detectionPoints[0];
            dStartPointX = origBitmap.getWidth() * detectionPoints[1];
            dEndPointY = origBitmap.getHeight() * detectionPoints[2];
            dEndPointX = origBitmap.getWidth() * detectionPoints[3];

            if (bestConfidence < this.detectionThreshold) {
                System.out.println("DETECTION DEBUG: Confidence under threshold. Setting output to EMPTY");
                imageClass = ImageClass.EMPTY;
            }

            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return new Detection(imageClass, new Point(dStartPointX, dStartPointY), new Point(dEndPointX, dEndPointY), bestConfidence);
    }


    public Detection classifyImageColorLarge(Mat image) {

        System.out.println("Detection with Color 640x640");

        ImageClass imageClass = ImageClass.EMPTY;

        float dStartPointX = 0;
        float dStartPointY = 0;
        float dEndPointX = 0;
        float dEndPointY = 0;

        float bestConfidence = 0;

        try {
            ColorLarge model = ColorLarge.newInstance(context);

            // Konvertoidaan Mat -> Bitmap

            Bitmap origBitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image, origBitmap);
            Bitmap bitmap = Bitmap.createScaledBitmap(origBitmap, imageSizeLarge, imageSizeLarge, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSizeLarge, imageSizeLarge, 3}, DataType.FLOAT32);

            // Työnnetään bitmap inputfeatureen
            inputFeature0.loadBuffer(getByteBuffer(bitmap));

            // Runs model inference and gets result.
            ColorLarge.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            float[] detectionPoints = outputFeature1.getFloatArray();
            int[] detections = outputFeature3.getIntArray();

            System.out.println("Confidences " + Arrays.toString(confidences));
            System.out.println("Detections " + Arrays.toString(detections));
            System.out.println("Detection points " + Arrays.toString(detectionPoints));

            int bestIndex = detections[0];
            bestConfidence = confidences[0];

            imageClass = ImageClass.values()[bestIndex];

            dStartPointY = origBitmap.getHeight() * detectionPoints[0];
            dStartPointX = origBitmap.getWidth() * detectionPoints[1];
            dEndPointY = origBitmap.getHeight() * detectionPoints[2];
            dEndPointX = origBitmap.getWidth() * detectionPoints[3];

            if (bestConfidence < this.detectionThreshold) {
                System.out.println("DETECTION DEBUG: Confidence under threshold. Setting output to EMPTY");
                imageClass = ImageClass.EMPTY;
            }

            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return new Detection(imageClass, new Point(dStartPointX, dStartPointY), new Point(dEndPointX, dEndPointY), bestConfidence);
    }


    public Detection classifyImageBWLarge(Mat image) {

        Mat image_bw = new Mat();
        Imgproc.cvtColor(image, image_bw, Imgproc.COLOR_BGR2GRAY);

        System.out.println("Detection with Grayscale 640x640");

        ImageClass imageClass = ImageClass.EMPTY;

        float dStartPointX = 0;
        float dStartPointY = 0;
        float dEndPointX = 0;
        float dEndPointY = 0;

        float bestConfidence = 0;

        try {
            BWLarge model = BWLarge.newInstance(context);

            // Konvertoidaan Mat -> Bitmap

            Bitmap origBitmap = Bitmap.createBitmap(image_bw.cols(), image_bw.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image_bw, origBitmap);
            Bitmap bitmap = Bitmap.createScaledBitmap(origBitmap, imageSizeLarge, imageSizeLarge, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSizeLarge, imageSizeLarge, 3}, DataType.FLOAT32);

            // Työnnetään bitmap inputfeatureen
            inputFeature0.loadBuffer(getByteBuffer(bitmap));

            // Runs model inference and gets result.
            BWLarge.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            float[] detectionPoints = outputFeature1.getFloatArray();
            int[] detections = outputFeature3.getIntArray();

            System.out.println("Confidences " + Arrays.toString(confidences));
            System.out.println("Detections " + Arrays.toString(detections));
            System.out.println("Detection points " + Arrays.toString(detectionPoints));

            int bestIndex = detections[0];
            bestConfidence = confidences[0];

            imageClass = ImageClass.values()[bestIndex];

            dStartPointY = origBitmap.getHeight() * detectionPoints[0];
            dStartPointX = origBitmap.getWidth() * detectionPoints[1];
            dEndPointY = origBitmap.getHeight() * detectionPoints[2];
            dEndPointX = origBitmap.getWidth() * detectionPoints[3];

            if (bestConfidence < this.detectionThreshold) {
                System.out.println("DETECTION DEBUG: Confidence under threshold. Setting output to EMPTY");
                imageClass = ImageClass.EMPTY;
            }

            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return new Detection(imageClass, new Point(dStartPointX, dStartPointY), new Point(dEndPointX, dEndPointY), bestConfidence);
    }

    public Detection classifyImageCDC(Mat image) {

        Detection init = classifyImageColorSmall(image);

        System.out.println("Detection with CDC");

        if (init.imgClass == ImageClass.EMPTY)
            return init;

        Mat croppedImg = image.submat((int) init.startPoint.y, (int) init.endPoint.y, (int) init.startPoint.x, (int) init.endPoint.x);

        Bitmap origBitmap = Bitmap.createBitmap(croppedImg.cols(), croppedImg.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(croppedImg, origBitmap);
        Bitmap bitmap = Bitmap.createScaledBitmap(origBitmap, 32, 32, false);

        try {
            Icmodel model = Icmodel.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 32 * 32 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[32 * 32];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < 32; i ++){
                for(int j = 0; j < 32; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Icmodel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            System.out.println("Confidences " + Arrays.toString(confidences));

            init.imgClass = ImageClass.values()[maxPos];

            // Releases model resources if no longer used.
            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return init;
    }
}

