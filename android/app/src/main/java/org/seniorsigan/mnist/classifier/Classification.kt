package org.seniorsigan.mnist.classifier

data class Classification(
        var conf: Float = -1f,
        var label: String? = null
) {
    val proba: String
        get() = "%.2f".format(conf)

    fun update(conf: Float, label: String) {
        this.conf = conf
        this.label = label
    }
}