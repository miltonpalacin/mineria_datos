/*
package pruebas

class test {
    private fun sendNotification(msg: String) {
        val intent = Intent(this, MainActivity::class.java)
        intent.putExtra("yourpackage.notifyId", NOTIFICATION_ID)
        val pIntent: PendingIntent = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT)
        val mBuilder: NotificationCompat.Builder = Builder(this)
                .setContentTitle("EXX")
                .setSmallIcon(R.drawable.ic_launcher)
                .setStyle(BigTextStyle()
                        .bigText(msg))
                .addAction(getNotificationIcon(), "Action Button", pIntent)
                .setContentIntent(pIntent)
                .setContentText(msg)
                .setOngoing(true)
        mNotificationManager.notify(NOTIFICATION_ID, mBuilder.build())
    }

    private fun sendNotification(msg: String) {
        val mBuilder: NotificationCompat.Builder = Builder(this)
                .setContentTitle("EXX")
                .setSmallIcon(R.drawable.ic_launcher)
                .setStyle(BigTextStyle()
                        .bigText(msg))
                .setContentText(msg)
                .setOngoing(true)
        setContentIntent(pendingIntent)
        mNotificationManager.notify(NOTIFICATION_ID, mBuilder.build())
    }
}*/

/*

package be.hcpl.android.sensors.service

import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import be.hcpl.android.sensors.service.SensorBackgroundService

*/
/*from ww  w  .ja va2s  .  c  om*//*
 */
/**
 * for a background service not linked to an activity it's important to use the command approach
 * instead of the Binder. For starting use the alarm manager
 *//*

class SensorBackgroundService : Service(), SensorEventListener {
    */
/**
     * again we need the sensor manager and sensor reference
     *//*

    private var mSensorManager: SensorManager? = null

    */
/**
     * an optional flag for logging
     *//*

    private var mLogging = false

    */
/**
     * treshold values
     *//*

    private var mThresholdMin = 0f
    private var mThresholdMax = 0f
    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {

        // get sensor manager on starting the service
        mSensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // have a default sensor configured
        var sensorType = Sensor.TYPE_LIGHT
        val args = intent.extras

        // get some properties from the intent
        if (args != null) {

            // set sensortype from bundle
            if (args.containsKey(KEY_SENSOR_TYPE)) sensorType = args.getInt(KEY_SENSOR_TYPE)

            // optional logging
            mLogging = args.getBoolean(KEY_LOGGING)

            // treshold values
            // since we want to take them into account only when configured use min and max
            // values for the type to disable
            mThresholdMin = if (args.containsKey(KEY_THRESHOLD_MIN_VALUE)) args.getFloat(
                    KEY_THRESHOLD_MIN_VALUE
            ) else Float.MIN_VALUE
            mThresholdMax = if (args.containsKey(KEY_THRESHOLD_MAX_VALUE)) args.getFloat(
                    KEY_THRESHOLD_MAX_VALUE
            ) else Float.MAX_VALUE
        }

        // we need the light sensor
        val sensor = mSensorManager!!.getDefaultSensor(sensorType)

        // TODO we could have the sensor reading delay configurable also though that won't do much
        // in this use case since we work with the alarm manager
        mSensorManager!!.registerListener(
                this, sensor,
                SensorManager.SENSOR_DELAY_NORMAL
        )
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder? {
        // ignore this since not linked to an activity
        return null
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // do nothing
    }

    override fun onSensorChanged(event: SensorEvent) {

        // for recording of data use an AsyncTask, we just need to compare some values so no
        // background stuff needed for this

        // Log that information for so we can track it in the console (for production code remove
        // this since this will take a lot of resources!!)
        if (mLogging) {

            // grab the values
            val sb = StringBuilder()
            for (value in event.values) sb.append(value.toString()).append(" | ")
            Log.d(
                    TAG,
                    "received sensor valures are: " + sb.toString() + " and previosValue was: " + previousValue
            )
        }

        // get the value
        // TODO we could make the value index also configurable, make it simple for now
        val sensorValue = event.values[0]

        // if first value is below min or above max threshold but only when configured
        // we need to enable the screen
        if (previousValue > mThresholdMin && sensorValue < mThresholdMin
                || previousValue < mThresholdMax && sensorValue > mThresholdMax
        ) {

            // and a check in between that there should have been a non triggering value before
            // we can mark a given value as trigger. This is to overcome unneeded wakeups during
            // night for instance where the sensor readings for a light sensor would always be below
            // the threshold needed for day time use.

            // TODO we could even make the actions configurable...

            // wake screen here
            val pm = applicationContext.getSystemService(POWER_SERVICE) as PowerManager
            val wakeLock = pm.newWakeLock(
                    PowerManager.SCREEN_BRIGHT_WAKE_LOCK or PowerManager.FULL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP,
                    TAG
            )
            wakeLock.acquire()

            //and release again
            wakeLock.release()

            // optional to release screen lock
            //KeyguardManager keyguardManager = (KeyguardManager) getApplicationContext().getSystemService(getApplicationContext().KEYGUARD_SERVICE);
            //KeyguardManager.KeyguardLock keyguardLock =  keyguardManager.newKeyguardLock(TAG);
            //keyguardLock.disableKeyguard();
        }
        previousValue = sensorValue

        // stop the sensor and service
        mSensorManager!!.unregisterListener(this)
        stopSelf()
    }

    companion object {
        */
/**
         * a tag for logging
         *//*

        private val TAG = SensorBackgroundService::class.java.simpleName

        */
/**
         * also keep track of the previous value
         *//*

        private var previousValue = 0f
        const val KEY_SENSOR_TYPE = "sensor_type"
        const val KEY_THRESHOLD_MIN_VALUE = "threshold_min_value"
        const val KEY_THRESHOLD_MAX_VALUE = "threshold_max_value"
        const val KEY_LOGGING = "logging"
    }
}*/
/*
package uni.signalcapture

import android.app.AlertDialog
import android.app.NotificationManager
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import java.util.*
import kotlin.concurrent.schedule

class MainActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        */
/* if (intent.getBooleanExtra("EXIT", false)) {
             //finish()
             finishAffinity()
             //System.exit(0);
            // super.onDestroy();
             //Process.killProcess(Process.myPid());
             //this.onDestroy()
             moveTaskToBack(true);
             exitProcess(0)

         }*//*


        setContentView(R.layout.activity_main)
        startService(Intent(this, Capture::class.java))


        val spinner: Spinner = findViewById(R.id.spn_list_action)
        // Crear adaptador para usar el array de string y asignarlo al spinner layout
        ArrayAdapter.createFromResource(
                this,
                R.array.spn_list_action,
                android.R.layout.simple_spinner_item
        ).also { adapter ->
            // Configurar el layout a usar cuando la lista aparece
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            // Aplicar el adaptador al spinner
            spinner.adapter = adapter
        }


    }

    fun onClickContinue(View: View)
    {
        val text = "El servicio captura sigue ejecutándose!"
        val duration = Toast.LENGTH_LONG

        val toast = Toast.makeText(applicationContext, text, duration)
        toast.show()

        //startActivity(intent)
        finishAffinity()
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun onClickExit(View: View)
    {

        val builder = AlertDialog.Builder(this)
        builder.setMessage("¿Desea cancelar la captura?")
                .setPositiveButton("Sí",
                        DialogInterface.OnClickListener { dialog, id ->
                            val service = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                            service.deleteNotificationChannel(Capture.CHANNEL_ID);
                            stopService(Intent(this, Capture::class.java))



                            //val intent = Intent(applicationContext, MainActivity::class.java)
                            */
/*intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP
                            intent.putExtra("EXIT", true)
                            startActivity(intent)*//*


                            */
/* intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
                             intent.putExtra("EXIT", true);
                             startActivity(intent);*//*

                            */
/*intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
                             intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK)
                             intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                             intent.putExtra("EXIT", true);*//*

                            //startActivity(intent);
                            */
/*finish()*//*

                            finishAffinity()
                            */
/*moveTaskToBack(true)
                            exitProcess(-1)*//*

                            Timer("SettingUp", false).schedule(1000) {
                                Process.killProcess(Process.myPid());
                            }


                            */
/* super.onDestroy();*//*

                        })
                .setNegativeButton("No",
                        DialogInterface.OnClickListener { dialog, id ->

                        })
        // Create the AlertDialog object and return it
        //builder.create()

        builder.show()

    }
}*/

/*  private fun isCaptureRunning(serviceClass: Class<*>): Boolean {
      val manager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
      for (service in manager.getRunningServices(Int.MAX_VALUE)) {
          if (serviceClass.name == service.service.className) {
              return true
          }
      }
      return false
  }*/
/* val mBuilder: NotificationCompat.Builder = Builder(this)
        .setContentTitle("EXX")
        .setSmallIcon(R.drawable.ic_launcher)
        .setStyle(Notification.BigTextStyle()
                .bigText(msg))
        .addAction(getNotificationIcon(), "Action Button", pIntent)
        .setContentIntent(pIntent)
        .setContentText(msg)
        .setOngoing(true)*/


/*
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


class Capture : Service(), SensorEventListener {
    */
/**
     * Sensor Manager: sensor de referencia
     *//*

    private var mSensorManager: SensorManager? = null

    */
/**
     * Log solo para debug
     *//*

    private var mLoggingDebug = false

    private var gravity = FloatArray(3)
    private var iniCount = -1

    */
/*
     * Inicio del servicio
     *//*

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {



        val channelId =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                    createNotificationChannel("CaptureAccelometer", "CaptureAccelometer")
                else
                    ""
        CHANNEL_ID = channelId

        val intent = Intent(this, MainActivity::class.java)
        val pIntent: PendingIntent = PendingIntent.getActivity(this, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT)

        val notification: Notification = Notification.Builder(this, channelId)
                .setContentTitle("Captura Acelerómetro para MCC607")
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setStyle(Notification.BigTextStyle()
                        .bigText("Presione para configurar"))
                .setContentText("Configurar")
                .setContentIntent(pIntent)
                .setOngoing(true)
                .build()
        startForeground(NOTIFICATION_ID, notification)


        // Obtener referencia al servicio de sensores
        mSensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // Foco en el acelerómetro
        val sensor = mSensorManager!!.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gravity = FloatArray(3)
        iniCount = 0
        //val sensor = mSensorManager!!.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        //linear acceleration = acceleration - acceleration due to gravity

        // Habilando el log en Debbug
        mLoggingDebug = true

        */
/*
        SENSOR_DELAY_FASTEST 0 microsecond - 18-20 ms
        SENSOR_DELAY_GAME 20,000 microsecond - 37-39 ms
        SENSOR_DELAY_UI 60,000 microsecond - 85-87 ms
        SENSOR_DELAY_NORMAL 200,000 microseconds - 224-225 ms
        *//*

        // in this use case since we work with the alarm manager
        mSensorManager!!.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL
        )
        return START_STICKY
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // do nothing
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun createNotificationChannel(channelId: String, channelName: String): String{
        val chan = NotificationChannel(channelId,
                channelName, NotificationManager.IMPORTANCE_NONE)
        chan.lightColor = Color.BLUE
        chan.lockscreenVisibility = Notification.VISIBILITY_PRIVATE
        val service = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        service.createNotificationChannel(chan)

        return channelId
    }

    override fun onSensorChanged(event: SensorEvent) {

        // alpha is calculated as t / (t + dT)
        // with t, the low-pass filter's time-constant
        // and dT, the event delivery rate

        val alpha = 0.8f
        val linearAcceleration = FloatArray(3)
        gravity[0] = alpha * gravity[0] + (1 - alpha) * event.values[0]
        gravity[1] = alpha * gravity[1] + (1 - alpha) * event.values[1]
        gravity[2] = alpha * gravity[2] + (1 - alpha) * event.values[2]
        linearAcceleration[0] = event.values[0] - gravity[0]
        linearAcceleration[1] = event.values[1] - gravity[1]
        linearAcceleration[2] = event.values[2] - gravity[2]

        if (iniCount != -1 && iniCount < 25)// Cinco Segundos
            iniCount++
        else {
            iniCount = -1
            Log.d(TAG, "onSensorChanged Timer" + event.timestamp + " X: " + linearAcceleration[0] + " Y: " + linearAcceleration[1] + " Z: " + linearAcceleration[2])
            //Log.d(TAG, "onSensorChanged Timer" + event.timestamp + " X: " + event.values[0] + " Y: " + event.values[1] + " Z: " + event.values[2])
        }

*/
/*        //mSensorManager!!.unregisterListener(this)
  *//*
      //stopSelf()
    }

    override fun onCreate() {
        super.onCreate()
        Toast.makeText(this, "Sevicio de captura iniciado", Toast.LENGTH_SHORT).show()
        isRunning = true
    }

    override fun onDestroy() {
        mSensorManager!!.unregisterListener(this)
        stopSelf()
        Toast.makeText(this, "Sevicio de captura finalizado", Toast.LENGTH_SHORT).show()
        isRunning = false
        super.onDestroy()
    }

    companion object {

        private val TAG = Capture::class.java.simpleName
        private val NOTIFICATION_ID = 2001
        public var CHANNEL_ID = ""
        public var isRunning = false
    }
}*/
