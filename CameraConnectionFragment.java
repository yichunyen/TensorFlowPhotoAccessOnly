/*
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.app.Fragment;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.Trace;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import junit.framework.Assert;

import java.io.IOException;
import java.util.List;

public class CameraConnectionFragment extends Fragment {
    public static CameraConnectionFragment newInstance() {
        return new CameraConnectionFragment();
    }

    @Override
    public View onCreateView(
            final LayoutInflater inflater, final ViewGroup container, final Bundle savedInstanceState) {
        return inflater.inflate(R.layout.camera_connection_fragment, container, false);
    }

    private TextView tvResult;
    private ImageView imvDisplaying;

    @Override
    public void onViewCreated(final View view, final Bundle savedInstanceState) {
        tvResult = (TextView) view.findViewById(R.id.tv_result);
        imvDisplaying = (ImageView) view.findViewById(R.id.imv_displaying);
        start();
    }

    @Override
    public void onActivityCreated(final Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    private static final int NUM_CLASSES = 1001;
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input:0";
    private static final String OUTPUT_NAME = "output:0";
    
//    private static final int NUM_CLASSES = 3;
//    private static final int INPUT_SIZE = 299;
//    private static final int IMAGE_MEAN = 128;
//    private static final float IMAGE_STD = 128;
//    private static final String INPUT_NAME = "Mul:0";
//    private static final String OUTPUT_NAME = "final_result:0";

    private void start() {
        System.out.print("Getting assets.");
        TensorFlowImageClassifier tensorflow = new TensorFlowImageClassifier();

        String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
        String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";

//        String MODEL_FILE = "file:///android_asset/tensorflow_flower_graph.pb";
//        String LABEL_FILE = "file:///android_asset/flowers_comp_graph_label_strings.txt";

        try {
            tensorflow.initializeTensorFlow(
                    getActivity().getAssets(), MODEL_FILE, LABEL_FILE, NUM_CLASSES, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD,
                    INPUT_NAME, OUTPUT_NAME);

            System.out.print("Tensorflow initialized.");

            Bitmap bitmap = BitmapFactory.decodeResource(getActivity().getResources(),
                    R.drawable.grace_hopper);

            imvDisplaying.setImageBitmap(bitmap);

            System.out.print("Initializing at size " + bitmap.getHeight() + " x " + bitmap.getWidth());
            Bitmap croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);

            drawResizedBitmap(bitmap, croppedBitmap);

            Trace.beginSection("imageAvailable");
            List<Classifier.Recognition> results = tensorflow.recognizeImage(croppedBitmap);

            System.out.print("results " + results.size());
            String strResult = "";
            for (Classifier.Recognition result : results) {
                strResult += result.getTitle() + " " + result.getConfidence() + "\n";
            }
            tvResult.setText(strResult);

        } catch (IOException exception) {
            System.out.print(exception.toString());
        }

    }

    private void drawResizedBitmap(final Bitmap src, final Bitmap dst) {
        Assert.assertEquals(dst.getWidth(), dst.getHeight());
        final float minDim = Math.min(src.getWidth(), src.getHeight());

        final Matrix matrix = new Matrix();

        // We only want the center square out of the original rectangle.
        final float translateX = -Math.max(0, (src.getWidth() - minDim) / 2);
        final float translateY = -Math.max(0, (src.getHeight() - minDim) / 2);
        matrix.preTranslate(translateX, translateY);

        final float scaleFactor = dst.getHeight() / minDim;
        matrix.postScale(scaleFactor, scaleFactor);

        // Rotate around the center if necessary.
//        if (sensorOrientation != 0) {
//            matrix.postTranslate(-dst.getWidth() / 2.0f, -dst.getHeight() / 2.0f);
//            matrix.postRotate(sensorOrientation);
//            matrix.postTranslate(dst.getWidth() / 2.0f, dst.getHeight() / 2.0f);
//        }

        final Canvas canvas = new Canvas(dst);
        canvas.drawBitmap(src, matrix, null);
    }
}
