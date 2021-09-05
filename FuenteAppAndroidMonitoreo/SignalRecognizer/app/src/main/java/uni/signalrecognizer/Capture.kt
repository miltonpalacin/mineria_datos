package uni.signalrecognizer

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import android.widget.Toast
import androidx.annotation.RequiresApi
import com.neovisionaries.ws.client.*
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.pow


class Capture : Service(), SensorEventListener {
    /**
     * Sensor Manager: sensor de referencia
     */
    private var mSensorManager: SensorManager? = null

    private var iniCount = -1
    private var sensorTimeReference = 0L
    private var myTimeReference = 0L
    private var WS: WebSocket? = null
    private var frecDelaySeg = 0
    private var unitTime = 10_000 //microseconds
    private var totalLapses = 0
    private var currentLapse = 1
    private var sendBegin = false
    private var FORMAT_TIME = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS")

    /*
     * Inicio del servicio
     */
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {

        val channelId =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                createNotificationChannel("MonitorAccelometer", "MonitorAccelometer")
            else
                ""
        CHANNEL_ID = channelId

        val intent = Intent(this, MainActivity::class.java)
        val pIntent: PendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification: Notification = Notification.Builder(this, channelId)
            .setContentTitle("Monitorear Actividad")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setStyle(
                Notification.BigTextStyle()
                    .bigText("Presione para configurar")
            )
            .setContentText("Configurar")
            .setContentIntent(pIntent)
            .setOngoing(true)
            .build()
        startForeground(NOTIFICATION_ID, notification)

        // Obtener referencia al servicio de sensores
        mSensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // Foco en el acelerómetro
        val mAccelerometer = mSensorManager!!.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        iniCount = 0
        sensorTimeReference = 0L
        myTimeReference = 0L

        GUID = UUID.randomUUID().toString()

        try {
            WS = WebSocketFactory().createSocket("ws://" + IP, 2000)

            WS?.addListener(object : WebSocketAdapter() {
                override fun onConnected(ws: WebSocket, headers: Map<String, List<String>>) {
                    Log.d("MONITOREO", "Connected")
                    //ws.sendText("Hello")
                    WS?.sendText("BEGIN,$GUID")
                }

                override fun onTextMessage(ws: WebSocket, text: String) {
                    Log.d("MONITOREO", "Message received: $text")
                    CURRENT_ACTION = text
                }

                override fun onConnectError(websocket: WebSocket?, exception: WebSocketException?) {
                    CURRENT_ACTION = "[{ERROR}]"
                }

            })

            WS?.connectAsynchronously()
            sendBegin = false

        } catch (e: Exception) {
            Toast.makeText(this, "Error: " + e.message, Toast.LENGTH_SHORT).show()
        }

        var delay = (10.0.pow(6.0) / FRECUENCY).toInt()

        //Esto debe ser igual a la frecuencia si es diferente de cero
        var frecuencia = (10.0.pow(6.0) / delay).toInt()
        frecDelaySeg = DELAY_CAPTURE * frecuencia

        Toast.makeText(
            this,
            "DELAY MICROSEG: $delay, FRECUENCIA: $frecuencia",
            Toast.LENGTH_SHORT
        ).show()
        totalLapses = delay / unitTime
        currentLapse = 1

        mSensorManager!!.registerListener(this, mAccelerometer, unitTime)

        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // do nothing
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun createNotificationChannel(channelId: String, channelName: String): String {
        val chan = NotificationChannel(
            channelId,
            channelName, NotificationManager.IMPORTANCE_NONE
        )
        chan.lightColor = Color.BLUE
        chan.lockscreenVisibility = Notification.VISIBILITY_PRIVATE
        val service = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        service.createNotificationChannel(chan)

        return channelId
    }


    //@RequiresApi(Build.VERSION_CODES.O)
    override fun onSensorChanged(event: SensorEvent) {

        if (currentLapse < totalLapses) {
            currentLapse++
            return
        }
        currentLapse = 1
        if (iniCount != -1 && iniCount < frecDelaySeg)// DELAY segundos
            iniCount++
        else {
            iniCount = -1

            /*if (!sendBegin) {
                if (WS?.isOpen == true)
                    WS?.sendText("BEGIN,$ACTION,$SEX,$AGE,$NAME")
                sendBegin = true
            }*/

            // Tiempos de referencia
            if (sensorTimeReference == 0L && myTimeReference == 0L) {
                sensorTimeReference = event.timestamp
                myTimeReference = System.currentTimeMillis()
            }
            // Timestamp en milisegundos
            var time =
                myTimeReference + Math.round((event.timestamp - sensorTimeReference) / 1000000.0)

            val times = FORMAT_TIME.format(time).split("-")
            val msg =
                "${times[1]},${times[2]},${times[3]},${times[4]},${times[5]},${times[6]},${event.values[0]},${event.values[1]},${event.values[2]},$GUID"

            if (LOG_DEBUG)
                Log.d(TAG, msg)

            if (WS?.isOpen == true)
                WS?.sendText(msg)
            else
                CURRENT_ACTION = ""


        }
    }

    override fun onCreate() {
        super.onCreate()
        Toast.makeText(this, "Sevicio de monitoreo iniciado", Toast.LENGTH_SHORT).show()
        IS_RUNNING = true
    }

    override fun onDestroy() {
        if (WS?.isOpen == true) {
            WS?.sendText("END," + GUID)
            WS?.disconnect(1000)
        }
        mSensorManager!!.unregisterListener(this)
        Toast.makeText(this, "Sevicio de monitoreo finalizado", Toast.LENGTH_SHORT).show()
        IS_RUNNING = false
        CURRENT_ACTION = ""
        stopSelf()
        super.onDestroy()

    }

    companion object {

        public val TAG = Capture::class.java.simpleName
        private val NOTIFICATION_ID = 2001
        public var CHANNEL_ID = ""
        public var IS_RUNNING = false
        public var LOG_DEBUG = true
        public var CURRENT_ACTION = ""
        public val FRECUENCY = 50
        public var IP = ""
        public var DELAY_CAPTURE = 3
        public var GUID = ""
    }
}