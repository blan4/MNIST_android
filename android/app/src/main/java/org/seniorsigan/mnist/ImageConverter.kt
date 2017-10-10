package org.seniorsigan.mnist

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.zomato.photofilters.imageprocessors.Filter
import com.zomato.photofilters.imageprocessors.subfilters.BrightnessSubfilter
import com.zomato.photofilters.imageprocessors.subfilters.ContrastSubfilter
import com.zomato.photofilters.imageprocessors.subfilters.SaturationSubfilter
import org.jetbrains.anko.doAsync

object ImageConverter {
    private val TAG = "MNIST - ImageConverter"

    init {
        System.loadLibrary("NativeImageProcessor")
    }

    fun normalize(pixels: IntArray): FloatArray {
        return pixels.map { pix ->
            val b = ImageUtils.getGreyColor(pix)
            val c = ((b / 255f) * -1f) + 1f // invert colors
            when {
                c < 0.2 -> 0f
                c < 0.65 -> 0f
                else -> 1f
            }
        }.toFloatArray()
    }

    /**
     * Get full picture, make it square, downscale, grayscale, so prepare!
     */
    fun prepare(bitmap: Bitmap, width: Int, context: Context, callback: (Bitmap) -> Unit) {
        val wh = if (bitmap.width > bitmap.height) {
            bitmap.height
        } else {
            bitmap.width
        }

        doAsync {
            val b = Bitmap.createScaledBitmap(bitmap, wh, wh, true).let {
                ImageUtils.getScaledDownBitmap(it, 500, false)
            }.let {
                ImageConverter.filters(it)
            }.let {
                ImageUtils.getScaledDownBitmap(it, width, false)
            }

            Log.i(TAG, "Scaled Photo shape: ${b.height}x${b.width}")


            callback(b)
        }
    }

    private fun filters(bitmap: Bitmap): Bitmap {
        val filter = Filter()
        filter.addSubFilter(BrightnessSubfilter(30))
        filter.addSubFilter(ContrastSubfilter(1.5f))
        filter.addSubFilter(SaturationSubfilter(1.3f))
        return filter.processFilter(bitmap)
    }
}