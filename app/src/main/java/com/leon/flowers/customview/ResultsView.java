package com.leon.flowers.customview;

import java.util.List;
import com.leon.flowers.tflite.Classifier.Recognition;

public interface ResultsView {
    public void setResults(final List<Recognition> results);
}
