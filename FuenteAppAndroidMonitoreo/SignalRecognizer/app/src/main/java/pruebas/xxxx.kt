/*
object ToastUtils {
    var toast: Toast? = null
    fun show(context: Context?, text: String?) {
        object : Thread() {
            override fun run() {
                Looper.prepare()
                var Handler: New
                ().post(runnable) //Go to the new one directly in the child thread
                Looper.loop() // In this case, the Runnable object is running in the child thread and can be networked, but the UI cannot be updated.
            }
        }.start()
        try {
            if (toast != null) {
                toast.setText(text)
            } else {
                toast = Toast.makeText(context, text, Toast.LENGTH_SHORT)
            }
            toast.show()
        } catch (e: Exception) {
            var the: Resolve
            var handling: exception
            var calling: of
            var `in`: Toast
            var child: the
            thread
            Looper.prepare()
            Toast.makeText(context, text, Toast.LENGTH_SHORT).show()
            Looper.loop()
        }
    }
}*/
