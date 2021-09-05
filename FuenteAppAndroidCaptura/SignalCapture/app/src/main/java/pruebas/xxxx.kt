/*
package com.shaikhhamadali.blogspot.playingwithbitmaps

import android.app.Activity
import android.graphics.*
import android.os.Bundle
import android.view.View
import android.widget.ImageView

class Main : Activity() {
    var imViewAndroid: ImageView? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imViewAndroid = findViewById<View>(R.id.imViewAndroid) as ImageView
        val p = Point()
        p[180] = 1000
        val b = waterMark(
            BitmapFactory.decodeResource(resources, R.drawable.car),
            "Welcome To Hamad's Blog",
            p,
            Color.WHITE,
            90,
            30,
            true
        )
        imViewAndroid!!.setImageBitmap(b)
    }

    fun waterMark(
        src: Bitmap,
        watermark: String?,
        location: Point,
        color: Int,
        alpha: Int,
        size: Int,
        underline: Boolean
    ): Bitmap {
        //get source image width and height
        val w = src.width
        val h = src.height
        val result = Bitmap.createBitmap(w, h, src.config)
        //create canvas object
        val canvas = Canvas(result)
        //draw bitmap on canvas
        canvas.drawBitmap(src, 0f, 0f, null)
        //create paint object
        val paint = Paint()
        //apply color
        paint.color = color
        //set transparency
        paint.alpha = alpha
        //set text size
        paint.textSize = size.toFloat()
        paint.isAntiAlias = true
        //set should be underlined or not
        paint.isUnderlineText = underline
        //draw text on given location
        canvas.drawText(watermark!!, location.x.toFloat(), location.y.toFloat(), paint)
        return result
    }
}*/
