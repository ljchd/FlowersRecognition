package com.leon.flowers.tflite;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class ClassifierMobileNetV2 extends Classifier {

    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 0f;

    private static final float IMAGE_STD = 255f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code ClassifierMobileNetV2}.
     *
     * @param activity
     */
    public ClassifierMobileNetV2(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    // TODO: Specify model.tflite as the model file and labels.txt as the label file
    @Override
    protected String getModelPath() {
        return "quantmobilenetv2flowers.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "labels.txt";
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
