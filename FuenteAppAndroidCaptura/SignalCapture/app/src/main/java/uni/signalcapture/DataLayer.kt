package uni.signalcapture

import android.os.Build
import android.os.Environment
import android.util.Log
import androidx.annotation.RequiresApi
import com.neovisionaries.ws.client.WebSocket
import java.io.File
import java.text.SimpleDateFormat
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

@RequiresApi(Build.VERSION_CODES.O)
class DataLayer(time: String, action: String, sex: String, age: String, name: String) {

    private val CSV_HEADER = "MM,dd,HH,mm,ss,sss,x,y,z,actividad,sexo,edad,nombre"
    private var BASE_FILE: File? = null

    //private var FILENAME = "captura_{estado}_{sexo}_{edad}_{date}.csv"
    private var FILENAME = "{action}_{sex}_{age}_{date}.csv"
    private var FORMAT_TIME =
        SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS")//DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss-SSS")//

    init {

        BASE_FILE = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        BASE_FILE = File(BASE_FILE, "CapturaMCC607")
        if (!BASE_FILE!!.exists())
            BASE_FILE!!.mkdirs()

        val current = LocalDateTime.now()

        var formatter = DateTimeFormatter.ofPattern("yyyyMMdd")

        if (time != "0")
            formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss")

        val formatted = current.format(formatter)
        FILENAME = (if (name.isNullOrEmpty()) "" else name.toLowerCase() + "_") + FILENAME
        FILENAME = FILENAME.replace("{action}", action.toLowerCase())
        FILENAME = FILENAME.replace("{sex}", sex.toLowerCase())
        FILENAME = FILENAME.replace("{age}", age.toLowerCase())
        FILENAME = FILENAME.replace("{date}", formatted)

        BASE_FILE = File(BASE_FILE, FILENAME)

    }

    public fun saveData(
        time: Long,
        data: FloatArray,
        action: String,
        sex: String,
        age: String,
        name: String,
        ws: WebSocket? = null
    ) {

        val times = FORMAT_TIME.format(time).split("-")
        val msg =
            "${times[1]},${times[2]},${times[3]},${times[4]},${times[5]},${times[6]},${data[0]},${data[1]},${data[2]},$action,$sex,$age,$name"
        if (Capture.LOG_DEBUG)
            Log.d(Capture.TAG, msg)

        if (ws?.isOpen == true)
            ws?.sendText(msg)
        else {
            if (!BASE_FILE!!.exists()) {
                BASE_FILE!!.createNewFile()
                BASE_FILE!!.appendText(CSV_HEADER)
                BASE_FILE!!.appendText("\n")
            }

            BASE_FILE?.appendText(msg)
            BASE_FILE?.appendText("\n")
        }
    }

}