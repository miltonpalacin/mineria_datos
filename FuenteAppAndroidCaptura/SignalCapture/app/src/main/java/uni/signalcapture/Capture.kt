package uni.signalcapture

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
import com.neovisionaries.ws.client.WebSocket
import com.neovisionaries.ws.client.WebSocketFactory
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

    @RequiresApi(Build.VERSION_CODES.O)
    private var dataLayer: DataLayer? = null

    /*
     * Inicio del servicio
     */
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {

        val channelId =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                createNotificationChannel("CaptureAccelometer", "CaptureAccelometer")
            else
                ""
        CHANNEL_ID = channelId

        val intent = Intent(this, MainActivity::class.java)
        val pIntent: PendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT
        )

        val notification: Notification = Notification.Builder(this, channelId)
            .setContentTitle("Captura Acelerómetro para MCC607")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setStyle(
                Notification.BigTextStyle()
                    .bigText("Presione para configurar - " + ACTION)
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

        /*
        SENSOR_DELAY_FASTEST 0 microsecond - 18-20 ms
        SENSOR_DELAY_GAME 20,000 microsecond - 37-39 ms
        SENSOR_DELAY_UI 60,000 microsecond - 85-87 ms
        SENSOR_DELAY_NORMAL 200,000 microseconds - 224-225 ms
        */

        try {
            WS = WebSocketFactory().createSocket("ws://" + IP, 3000)
            WS?.connectAsynchronously()
            sendBegin = false

        } catch (e: Exception) {
            Toast.makeText(this, "Error: " + e.message, Toast.LENGTH_SHORT).show()
        }

        // Iniciar DataLayer
        dataLayer = DataLayer(TIME, ACTION, SEX, AGE, NAME)

        var delay =
            (if (FRECUENCY.isNullOrEmpty() || FRECUENCY == "0") 20000.0
            else 10.0.pow(6.0) / FRECUENCY.toFloat()).toInt()

        //Esto debe ser igual a la frecuencia si es diferente de cero
        var frecuencia = (10.0.pow(6.0) / delay).toInt()
        TOTAL_CICLO =
            frecuencia * (if (TIME.isNullOrEmpty() || TIME == "0") 24 * 60 * 60 else TIME.toInt())
        frecDelaySeg = DELAY_CAPTURE * frecuencia
        ACTUAL_CICLO = 0

        Toast.makeText(
            this,
            "DELYA MICROSEG: $delay,CICLOS TOTALES: $TOTAL_CICLO, FRECUENCIA: $frecuencia",
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


    @RequiresApi(Build.VERSION_CODES.O)
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

            if (!sendBegin) {
                if (WS?.isOpen == true)
                    WS?.sendText("BEGIN,$ACTION,$SEX,$AGE,$NAME")
                sendBegin = true
            }

            // Tiempos de referencia
            if (sensorTimeReference == 0L && myTimeReference == 0L) {
                sensorTimeReference = event.timestamp
                myTimeReference = System.currentTimeMillis()
            }
            // Timestamp en milisegundos
            var time =
                myTimeReference + Math.round((event.timestamp - sensorTimeReference) / 1000000.0)

            if (ACTUAL_CICLO < TOTAL_CICLO) {
                dataLayer?.saveData(time, event.values, ACTION, SEX, AGE, NAME, WS)
                ACTUAL_CICLO++
            } else
                onDestroy()

        }
    }

    override fun onCreate() {
        super.onCreate()
        Toast.makeText(this, "Sevicio de captura iniciado", Toast.LENGTH_SHORT).show()
        IS_RUNNING = true
    }

    override fun onDestroy() {
        if (WS?.isOpen == true) {
            WS?.sendText("END," + NAME)
            WS?.disconnect(1000)
        }
        mSensorManager!!.unregisterListener(this)
        Toast.makeText(this, "Sevicio de captura finalizado", Toast.LENGTH_SHORT).show()
        IS_RUNNING = false
        //TOTAL_CICLO = 0
        //ACTUAL_CICLO = 0
        NAME = ""
        stopSelf()
        super.onDestroy()

    }

    companion object {

        public val TAG = Capture::class.java.simpleName
        private val NOTIFICATION_ID = 2001
        public var CHANNEL_ID = ""
        public var IS_RUNNING = false
        public var LOG_DEBUG = true
        public var ACTION = ""
        public var SEX = ""
        public var AGE = ""
        public var TIME = ""
        public var FRECUENCY = ""
        public var IP = ""
        public var TOTAL_CICLO: Int = 0
        public var ACTUAL_CICLO: Int = 0
        public var DELAY_CAPTURE = 3
        public var NAME = ""
    }
}