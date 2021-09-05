package uni.signalrecognizer

import android.app.AlertDialog
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.os.*
import android.view.View
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.*
import kotlin.concurrent.schedule


class MainActivity : AppCompatActivity() {

    private var btnSS: Button? = null
    private var btnHide: Button? = null
    private var edtIp: EditText? = null
    private var txvState: TextView? = null
    private var lastState = ""
    private var stateTimer: Timer? = null
    private var finishBtnPressed = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        btnSS = findViewById(R.id.btn_start_stop)
        btnHide = findViewById(R.id.btn_hide)
        edtIp = findViewById(R.id.edt_ip)
        txvState = findViewById(R.id.txv_state)

        lastState = R.string.txv_state.toString()

        if (!Capture.IP.isNullOrEmpty())
            edtIp?.setText(Capture.IP)

        if (Capture.IS_RUNNING) {
            btnSS?.text = "¡Parar monitoreo!"
            btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.parar_servicio))
            btnHide?.visibility = View.VISIBLE
            edtIp?.isEnabled = false
            onState()

        } else {
            btnSS?.text = "¡Empezar monitoreo!"
            btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.rojo_uni))
            edtIp?.isEnabled = true
        }

        /*if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
        )
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_WRITE_STORAGE
            );*/


        /*val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)*/
    }

    fun onClickHide(ViewBtn: View) {
        finishAffinity()
    }

    fun onClickStartStop(ViewBtn: View) {

        if (Capture.IS_RUNNING) {
            stopService()

        } else {
            btnSS?.text = "¡Parar monitoreo!"
            btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.parar_servicio))
            Toast.makeText(
                this,
                "¡El servicio captura se envío a ejecutar!",
                Toast.LENGTH_SHORT
            ).show()
            edtIp?.isEnabled = false
            Capture.IP = edtIp?.text.toString()
            Capture.IS_RUNNING = true
            startService(Intent(this, Capture::class.java))
            btnHide?.visibility = View.VISIBLE

            onState()
        }
    }

    private fun stopService() {
        finishBtnPressed = true
        btnSS?.text = "¡Empezar monitoreo!"
        btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.rojo_uni))

        Toast.makeText(
            this, "¡El servicio de monitoreo se envío a parar!", Toast.LENGTH_SHORT
        ).show()

        stopService(Intent(this, Capture::class.java))
        stateTimer?.cancel()
        btnHide?.visibility = View.GONE
        edtIp?.isEnabled = true
        txvState?.setText(R.string.txv_state)
        lastState = R.string.txv_state.toString()
    }

    override fun onResume() {
        super.onResume()
        Toast.makeText(
            this,
            "VOLVÍ",
            Toast.LENGTH_SHORT
        ).show()

        if (!Capture.IS_RUNNING) txvState?.setText(R.string.txv_state)
    }

    private fun onState() {
        finishBtnPressed = false

        object : CountDownTimer(24 * 60 * 60 * 1000L, 2000) {
            override fun onTick(millisUntilFinished: Long) {
                if (finishBtnPressed) {
                    stopService()
                    cancel()
                }

                if (!Capture.CURRENT_ACTION.isNullOrEmpty()) {
                    if (Capture.CURRENT_ACTION == "[{ERROR}]")
                        stopService()
                    else
                        txvState?.text = Capture.CURRENT_ACTION
                }

            }

            override fun onFinish() {
                if (!finishBtnPressed) {
                    start();
                } else {
                    stopService()
                }
            }
        }.start()

    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun onClickExit(View: View) {

        val builder = AlertDialog.Builder(this)
        builder.setMessage("¿Desea cancelar la monitoreo?")
            .setPositiveButton(
                "Sí"
            ) { _, id ->
                val service =
                    getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                service.deleteNotificationChannel(Capture.CHANNEL_ID);
                stopService(Intent(this, Capture::class.java))
                finishAffinity()
                Timer("SettingUp", false).schedule(2000) {
                    Process.killProcess(Process.myPid());
                }
            }
            .setNegativeButton("No", { dialog, id -> })
        builder.show()
    }

    companion object {
        const val REQUEST_WRITE_STORAGE = 112
        //const val REQUEST_READ_STORAGE = 113
    }

}

