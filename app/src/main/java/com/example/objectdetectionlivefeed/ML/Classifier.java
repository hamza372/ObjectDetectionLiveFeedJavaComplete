package com.example.objectdetectionlivefeed.ML;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static java.lang.Math.min;

public class Classifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE ;
    private int PIXEL_SIZE = 3;
    private int IMAGE_MEAN = 0;
    private float IMAGE_STD = 127.0f;
    private boolean quantized;
    private int MAX_POSSIBLE_DETECTIONS = 10;
    public Classifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize, boolean quantized) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);
        this.quantized = quantized;
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelPath);
    }

    public class Recognition {

        private final String id;
        private final String title;
        private final Float confidence;
        private RectF location;

        public Recognition(final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }


    private MappedByteBuffer loadModelFile(AssetManager assetManager, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    private float[][][] outputLocations;
    private float[][] outputClasses;
    private float[][] outputScores;
    private float[] numDetections;
    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }


    public List<Recognition> recognizeImage(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        outputLocations = new float[1][MAX_POSSIBLE_DETECTIONS][4];
        outputClasses = new float[1][MAX_POSSIBLE_DETECTIONS];
        outputScores = new float[1][MAX_POSSIBLE_DETECTIONS];
        numDetections = new float[1];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        Object[] inputArray = {byteBuffer};
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        int numDetectionsOutput = min(MAX_POSSIBLE_DETECTIONS, (int) numDetections[0]);
        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            final RectF detection = new RectF(
                    outputLocations[0][i][1]*bitmap.getWidth(),
                    outputLocations[0][i][0]*bitmap.getHeight(),
                    outputLocations[0][i][3]*bitmap.getWidth(),
                    outputLocations[0][i][2]*bitmap.getHeight());
            recognitions.add(new Recognition("" + i, labelList.get((int) outputClasses[0][i]), outputScores[0][i], detection));
        }
        return  recognitions;
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        if(quantized){
            byteBuffer = ByteBuffer.allocateDirect(1  * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        }else{
            byteBuffer = ByteBuffer.allocateDirect(4  * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                if(quantized){
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                }else{
                    byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }
            }
        }
        return byteBuffer;
    }


    public void close(){
        interpreter.close();
    }

}

