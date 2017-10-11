package org.seniorsigan.mnist.classifier

import android.content.res.AssetManager
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class TensorFlowAutoencoder(
        assetManager: AssetManager,
        modelPath: String,
        val inputSize: Long,
        val inputName: String,
        val outputName: String
) {
    private val TAG = "TensorFlow"
    private var tfHelper = TensorFlowInferenceInterface(assetManager, modelPath)
    private var outputNames = arrayOf(outputName)
    private var output = FloatArray((inputSize*inputSize).toInt())

    fun transform(pixels: FloatArray): FloatArray {

        //using the interface
        //give it the input name, raw pixels from the drawing,
        //input size
        tfHelper.feed(inputName, pixels, 1L, inputSize, inputSize, 1L)

        //get the possible outputs
        tfHelper.run(outputNames)

        //get the output
        tfHelper.fetch(outputName, output)

        return output
    }
}