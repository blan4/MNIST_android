package org.seniorsigan.mnist.classifier

import android.content.res.AssetManager
import android.util.Log
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class TensorFlowClassifier(
        assetManager: AssetManager,
        override val name: String,
        modelPath: String,
        val inputSize: Long,
        val inputName: String,
        val outputName: String,
        val feedKeepProb: Boolean
) : Classifier {
    private val TAG = "TensorFlow"
    private val THRESHOLD = 0.1f
    private var tfHelper = TensorFlowInferenceInterface(assetManager, modelPath)
    private val numClasses = 10
    private var labels = 0.rangeTo(10).map { it.toString() }
    private var output = FloatArray(numClasses)
    private var outputNames = arrayOf(outputName)

    override fun recognize(pixels: FloatArray): Classification {

        //using the interface
        //give it the input name, raw pixels from the drawing,
        //input size
        tfHelper.feed(inputName, pixels, 1L, inputSize, inputSize, 1L)

        //probabilities
        if (feedKeepProb) {
            tfHelper.feed("keep_prob", floatArrayOf(1f))
        }
        //get the possible outputs
        tfHelper.run(outputNames)

        //get the output
        tfHelper.fetch(outputName, output)

        // Find the best classification
        //for each output prediction
        //if its above the threshold for accuracy we predefined
        //write it out to the view
        val ans = Classification()
        for (i in output.indices) {
            Log.d(TAG, "Out: ${output[i]}, Label: ${labels[i]}")
            if (output[i] > THRESHOLD && output[i] > ans.conf) {
                ans.update(output[i], labels[i])
            }
        }

        return ans
    }
}