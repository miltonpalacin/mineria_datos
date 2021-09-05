package uni.signalcapture

import android.Manifest
import android.app.AlertDialog
import android.app.NotificationManager
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.os.Process
import android.view.View
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.*
import kotlin.concurrent.schedule
import kotlin.math.pow


class MainActivity : AppCompatActivity() {

    private var spnAction: Spinner? = null
    private var spnSex: Spinner? = null
    private var spnAge: Spinner? = null
    private var btnSS: Button? = null
    private var btnHide: Button? = null
    private var edtTime: EditText? = null
    private var edtFrecuency: EditText? = null
    private var edtIp: EditText? = null
    private var txvNumSeg: TextView? = null
    private var edtName: EditText? = null
    private var counter = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)


        spnAction = findViewById<Spinner>(R.id.spn_list_action)
        spnSex = findViewById<Spinner>(R.id.spn_list_sex)
        spnAge = findViewById<Spinner>(R.id.spn_list_age)
        btnSS = findViewById<Button>(R.id.btn_start_stop)
        btnHide = findViewById<Button>(R.id.btn_hide)
        edtTime = findViewById<EditText>(R.id.edt_time)
        edtFrecuency = findViewById<EditText>(R.id.edt_frecuency)
        edtIp = findViewById<EditText>(R.id.edt_ip)
        txvNumSeg = findViewById<TextView>(R.id.txv_num_seg)
        edtName = findViewById<EditText>(R.id.edt_name)

        var selAction = 0
        var selSex = 0
        var selAge = 0

        var adapter_action = ArrayAdapter.createFromResource(
            this,
            R.array.spn_list_action,
            android.R.layout.simple_spinner_item
        )
        adapter_action.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spnAction!!.adapter = adapter_action

        var adapter_sex = ArrayAdapter.createFromResource(
            this,
            R.array.spn_list_sex,
            android.R.layout.simple_spinner_item
        )
        adapter_sex.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spnSex!!.adapter = adapter_sex

        var adapter_age = ArrayAdapter.createFromResource(
            this,
            R.array.spn_list_age,
            android.R.layout.simple_spinner_item
        )
        adapter_age.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spnAge!!.adapter = adapter_age

        if (!Capture.ACTION.isNullOrEmpty()) {
            selAction = adapter_action.getPosition(Capture.ACTION)
            selSex = adapter_sex.getPosition(Capture.SEX)
            selAge = adapter_sex.getPosition(Capture.AGE)
            spnAction?.setSelection(selAction)
            spnSex?.setSelection(selSex)
            spnAge?.setSelection(selAge)
            edtTime?.setText(Capture.TIME)
            edtFrecuency?.setText(Capture.FRECUENCY)
            edtIp?.setText(Capture.IP)
            edtName?.setText(Capture.NAME)
        }

        if (Capture.IS_RUNNING) {
            btnSS?.text = "¡Parar captura!"
            btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.parar_servicio))
            btnHide?.visibility = View.VISIBLE
            spnAction?.isEnabled = false
            spnSex?.isEnabled = false
            spnAge?.isEnabled = false
            edtTime?.isEnabled = false
            edtFrecuency?.isEnabled = false
            edtIp?.isEnabled = false
            edtName?.isEnabled = false

            onCounter()

        } else {
            btnSS?.text = "¡Empezar captura!"
            btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.rojo_uni))
            spnAction?.isEnabled = true
            spnSex?.isEnabled = true
            spnAge?.isEnabled = true
            edtTime?.isEnabled = true
            edtFrecuency?.isEnabled = true
            edtIp?.isEnabled = true
            edtName?.isEnabled = true
        }

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
        )
            ActivityCompat.requestPermissions(
                this,
                arrayOf<String>(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_WRITE_STORAGE
            );

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
            btnSS?.text = "¡Parar captura!"
            btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.parar_servicio))
            Toast.makeText(
                applicationContext,
                "¡El servicio captura se envío a ejecutar!",
                Toast.LENGTH_SHORT
            ).show()
            spnAction?.isEnabled = false
            spnSex?.isEnabled = false
            spnAge?.isEnabled = false
            edtTime?.isEnabled = false
            edtFrecuency?.isEnabled = false
            edtIp?.isEnabled = false
            edtName?.isEnabled = false
            Capture.ACTION = spnAction?.selectedItem.toString()
            Capture.SEX = spnSex?.selectedItem.toString()
            Capture.AGE = spnAge?.selectedItem.toString()
            Capture.TIME = edtTime?.text.toString()
            Capture.FRECUENCY = edtFrecuency?.text.toString()
            Capture.IP = edtIp?.text.toString()
            Capture.NAME = edtName?.text.toString()
            Capture.ACTUAL_CICLO = 0
            Capture.IS_RUNNING = true
            txvNumSeg?.setText("0")
            startService(Intent(this, Capture::class.java))
            btnHide?.visibility = View.VISIBLE

            onCounter()
        }
    }

    private fun stopService() {
        btnSS?.text = "¡Empezar captura!"
        btnSS?.setBackgroundColor(ContextCompat.getColor(this, R.color.rojo_uni))
        Toast.makeText(
            applicationContext,
            "¡El servicio captura se envío a parar!",
            Toast.LENGTH_SHORT
        ).show()
        stopService(Intent(this, Capture::class.java))
        spnAction?.isEnabled = true
        spnSex?.isEnabled = true
        spnAge?.isEnabled = true
        btnHide?.visibility = View.GONE
        edtTime?.isEnabled = true
        edtFrecuency?.isEnabled = true
        edtIp?.isEnabled = true
        edtName?.isEnabled = true
    }

    override fun onResume() {
        super.onResume()
        Toast.makeText(
            this,
            "VOLVÍ",
            Toast.LENGTH_SHORT
        ).show()

        if (Capture.IS_RUNNING) {
            var delay =
                (if (Capture.FRECUENCY.isNullOrEmpty() || Capture.FRECUENCY == "0") 20000.0
                else 10.0.pow(6.0) / Capture.FRECUENCY.toFloat()).toInt()
            var frecuencia = (10.0.pow(6.0) / delay).toInt()
            if (Capture.ACTUAL_CICLO != 0)
                counter = Capture.ACTUAL_CICLO / frecuencia + 1
            txvNumSeg?.setText((counter - 1).toString())
        }
        else
            txvNumSeg?.setText((counter).toString())

    }

    private fun onCounter() {

        counter = 1

        var delay =
            (if (Capture.FRECUENCY.isNullOrEmpty() || Capture.FRECUENCY == "0") 20000.0
            else 10.0.pow(6.0) / Capture.FRECUENCY.toFloat()).toInt()

        var frecuencia = (10.0.pow(6.0) / delay).toInt()
        var totalCiclo =
            frecuencia * (if (Capture.TIME.isNullOrEmpty() || Capture.TIME == "0") 24 * 60 * 60 else Capture.TIME.toInt())

        var delayCapture = Capture.DELAY_CAPTURE
        if (Capture.ACTUAL_CICLO != 0) {
            counter = Capture.ACTUAL_CICLO / frecuencia + 1
            delayCapture = -1
        }

        var totalSeg = totalCiclo / frecuencia

        object : CountDownTimer(1000L * (totalCiclo - Capture.ACTUAL_CICLO) / frecuencia, 1000) {
            override fun onTick(millisUntilFinished: Long) {

                if (delayCapture > 0)
                    --delayCapture
                else if (delayCapture == 0) {
                    counter = 1
                    delayCapture = -1
                }

                if (!Capture.IS_RUNNING) {
                    //Capture.ACTUAL_CICLO = 0
                    stopService()
                    cancel()
                }

                txvNumSeg?.setText(counter.toString())
                if (counter < totalSeg) counter++

            }

            override fun onFinish() {
                if (Capture.ACTUAL_CICLO < Capture.TOTAL_CICLO)
                    start()
                else {
                    //counter = -1
                    //Capture.ACTUAL_CICLO = 0
                    //txvNumSeg?.setText("0")
                    stopService()
                }
            }
        }.start()

    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun onClickExit(View: View) {

        val builder = AlertDialog.Builder(this)
        builder.setMessage("¿Desea cancelar la captura?")
            .setPositiveButton("Sí",
                DialogInterface.OnClickListener { dialog, id ->
                    val service =
                        getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                    service.deleteNotificationChannel(Capture.CHANNEL_ID);
                    stopService(Intent(this, Capture::class.java))
                    finishAffinity()
                    Timer("SettingUp", false).schedule(2000) {
                        Process.killProcess(Process.myPid());
                    }
                })
            .setNegativeButton("No", DialogInterface.OnClickListener { dialog, id -> })
        builder.show()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            REQUEST_WRITE_STORAGE -> {
                // If request is cancelled, the result arrays are empty.
                if ((grantResults.isNotEmpty() &&
                            grantResults[0] == PackageManager.PERMISSION_GRANTED)
                ) {

                } else {
                    val builder = AlertDialog.Builder(this)
                    builder.setMessage(
                        "Permisos para acceder al ALMACENAMIENTO es requerido para guardar las capturas," +
                                " Tendrá que configurarlo de manera manual en configuraciones de las Aplicaciones Instaladas."
                    )
                        .setTitle("Permiso requerido")

                    val dialog = builder.create()
                    dialog.show()
                }
                return
            }

            else -> {
                super.onRequestPermissionsResult(requestCode, permissions, grantResults)
            }
        }
    }

    companion object {
        const val REQUEST_WRITE_STORAGE = 112
        //const val REQUEST_READ_STORAGE = 113
    }

}

