package org.seniorsigan.mnist.classifier

interface Classifier {
    val name: String
    fun recognize(pixels: FloatArray): Classification
}