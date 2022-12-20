package com.example.seetrafficsignsopencv;

/*
This software project uses MIT -license and uses 3rd party parts that are licensed via Apache License 2.0.
--------
Copyright 2022 Solita Oy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import android.widget.Button;
import android.widget.ImageView;

import androidx.preference.PreferenceManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {

    private ImageClassifier imageClassifier;

    private final static String LOGTAG = "OpenCV_Log";
    private CameraBridgeViewBase mOpenCvCameraView;

    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.v(LOGTAG, "OpenCV loaded");
                mOpenCvCameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }

        }
    };

    Button btn_setting, btn_camera, btn_exit;
    Boolean debug_mode;
    String model;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        debug_mode = (prefs.getBoolean("debug_mode", true));
        model = (prefs.getString("models","CDC"));
        Log.d("MODEL",model);



        imageClassifier = new ImageClassifier(this);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener);

        btn_setting = findViewById(R.id.btn_setting);
        btn_camera = (Button) findViewById(R.id.btn_camera);

        btn_exit = findViewById(R.id.btn_exit);

        btn_setting.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
            startActivity(intent);
        });

        btn_camera.setOnClickListener(view -> {
            if (mOpenCvCameraView.getAlpha() == 0) {
                mOpenCvCameraView.setAlpha(1);
            } else {
                mOpenCvCameraView.setAlpha(0);
            }
       });

        btn_exit.setOnClickListener(view -> {
            moveTaskToBack(true);
            android.os.Process.killProcess(android.os.Process.myPid());
            System.exit(1);
        });

    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    private final CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener = new CameraBridgeViewBase.CvCameraViewListener2() {
        @Override
        public void onCameraViewStarted(int width, int height) {

        }

        @Override
        public void onCameraViewStopped() {

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

            try {

                Mat input_rgba = inputFrame.rgba();

                Detection detection;

                System.out.println("Detection with: " + model);

                switch (model) {
                    case "640BW":
                        detection = imageClassifier.classifyImageBWLarge(input_rgba);
                        break;
                    case "640C":
                        detection = imageClassifier.classifyImageColorLarge(input_rgba);
                        break;
                    case "CDC":
                        detection = imageClassifier.classifyImageCDC(input_rgba);
                        break;
                    case "320BW":
                        detection = imageClassifier.classifyImageBWSmall(input_rgba);
                        break;
                    case "320C":
                        detection = imageClassifier.classifyImageColorSmall(input_rgba);
                        break;
                    case "320C_1.1":
                        detection = imageClassifier.classifyImageColorSmallOld(input_rgba);
                        break;
                    default:
                        detection = imageClassifier.classifyImage(input_rgba);
                        break;
                }

                ImageClass frameClass = detection.imgClass;

                ImageView speedImage = (ImageView) findViewById(R.id.SLDisplay);
                runOnUiThread(() -> {
                    if (frameClass != ImageClass.EMPTY) {
                        speedImage.setImageResource(frameClass.id());
                    }
                });

                if (frameClass != ImageClass.EMPTY && debug_mode) {
                    Imgproc.rectangle(input_rgba, detection.startPoint, detection.endPoint, new Scalar(255, 222, 0), 3);

                    int confidence = (int) (detection.confidence * 100 + 0.5);

                    Imgproc.putText(input_rgba, frameClass.toString() + " " + confidence + " %", new Point(10, 50),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(255, 222, 0), 2, Imgproc.LINE_AA, false);
                }
                return input_rgba;
            }
            catch (Exception e){
                return inputFrame.rgba();
            }
        }
    };


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(LOGTAG, "OpenCV not found");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }
}