# Servidor WEBSOCKET

"""

asyncio
-------
Es una librería para escribir código concurrente usando
sintaxis como async/wait (invocación asíncrona).

wesockets
---------
Es una librería para construir servidores y clientes WebSockets, 
enfocado a la simplicidad y al funcionamiento correcto. Construido 
en base a la librería asyncio.

"""

import asyncio
import websockets
from datetime import datetime
import traceback


import numpy as np
#import pandas as pd
import joblib
import os
from scipy.ndimage import gaussian_filter1d
import numpy.fft as fft
import math
from sklearn.preprocessing import Normalizer
from scipy import signal
import warnings
#warnings.filterwarnings('ignore')

def spectral_centroid(x, spect_x):
    magnitudes = np.abs(spect_x)
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length)[:length//2 + 1])
    magnitudes = magnitudes[:length//2 + 1]
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)

def magnitude(x, y, z):
    l = len(x)
    sum = 0
    for i in range(l):
        sum = sum + math.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
    return sum/l

def obtener_muestra(data_ws):

    lb_x = gaussian_filter1d(data_ws[:, 0], 1)
    lb_y = gaussian_filter1d(data_ws[:, 1], 1)
    lb_z = gaussian_filter1d(data_ws[:, 2], 1)

    spect_x = abs(fft.fft(lb_x))
    spect_y = abs(fft.fft(lb_y))
    spect_z = abs(fft.fft(lb_z))

    spectral_centroid_x = spectral_centroid(lb_x, spect_x)
    spectral_centroid_y = spectral_centroid(lb_y, spect_y)
    spectral_centroid_z = spectral_centroid(lb_z, spect_z)

    magnit = magnitude(lb_x, lb_y, lb_z)
    adm_x = np.average(abs(lb_x - magnit))
    adm_y = np.average(abs(lb_y - magnit))
    adm_z = np.average(abs(lb_z - magnit))

    magnitSpect = magnitude(spect_x, spect_y, spect_z)
    admSpect_x = np.average(abs(spect_x - magnitSpect))
    admSpect_y = np.average(abs(spect_y - magnitSpect))
    admSpect_z = np.average(abs(spect_z - magnitSpect))

    row = []
    row.append(np.mean(lb_x)) # media
    row.append(np.mean(lb_y)) # media
    row.append(np.mean(lb_z)) # media
    row.append(np.mean(spect_x)) # media
    row.append(np.mean(spect_y)) # media
    row.append(np.mean(spect_z)) # media
    row.append(np.median(lb_x)) # mediana
    row.append(np.median(lb_y)) # mediana
    row.append(np.median(lb_z)) # mediana
    row.append(np.median(spect_x)) # mediana
    row.append(np.median(spect_y)) # mediana
    row.append(np.median(spect_z)) # mediana
    row.append(spectral_centroid_x) # spectral centroid
    row.append(spectral_centroid_y) # spectral centroid
    row.append(spectral_centroid_z) # spectral centroid
    row.append(magnit) # magnitude
    row.append(adm_x) # Average Difference from Mean
    row.append(adm_y) # Average Difference from Mean
    row.append(adm_z) # Average Difference from Mean
    row.append(magnitSpect) # magnitude spect
    row.append(admSpect_x) # Average Difference from Mean Spect
    row.append(admSpect_y) # Average Difference from Mean Spect
    row.append(admSpect_z) # Average Difference from Mean Spect
    return [row]

async def procesar_modelo(sample):
    global model

    X = obtener_muestra(sample)
    numEstado = model.predict(X)[0]
   
    estado = "DESCONOCIDO"
    # np.where(ds["class"] == "caminando", 0, np.where(ds["class"] == "sentado", 1, 2))
    if numEstado == 0:
        estado = "CAMINANDO"
    elif numEstado == 1:
        estado = "SENTADO"
    elif numEstado == 2:
        estado = "SALTANDO"
    elif numEstado == 3:
        estado = "CORRIENDO"
    else:
        estado = "SUBIENDO"

    return estado

sample_global = {}

# Función que procesa el mensaje del websocket
# en este caso recibe la señal de la conexión que realiza el celular
async def message_process(websocket, message):

    global sample_global

    code = message[message.rfind(",") + 1 : ]
    
    # IMPO await websocket.send("greeting")
    if message.startswith("BEGIN,"):
        
        if code not in sample_global:
            sample_global[code] = []

        print(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"Monitoreo iniciado de:", code)

    elif message.startswith("END,"):    
        # Finalizar captura
        print(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"Monitoreo finalizado de:", code)
    else:

        # Generar el vector con los valores de X, Y y Z
        data = message.split(",")
        row = [float(data[6]), float(data[7]), float(data[8])]
        
        # Acumular las capturas que corresponden al GUID
        sample_global[code].append(row)

        # 100 el equivalente a 2 segundos, debido a la frecuencia de muestreo igual a 5oHz
        # además en la muestras se considera lo mismo por el procedimiento de "Data Augmentation"
        if len(sample_global[code]) >= 100: 
            estado = await  procesar_modelo(np.array(sample_global[code]))
            sample_global[code] = []
            print(code, estado)
            # TO-DO, falta enviar el GUID
            await websocket.send(estado)
   
# Función controlador de conexiones:
# el presenta controlador solo se encargará de recibir la señal
# no responderá (enviará) mensajes de respuesta o feedback al cliente.
async def signal_capture(websocket, path):

    print("Websocket iniciado")
    # Procesar los mensajes de forma continua sin cerrar 
    # la conexión con los clientes
    try:
        async for message in websocket:
            #print(message)
            await message_process(websocket, message)
    except Exception as e:
        print ("Ha ocurrido un error:", e)
        print (traceback.format_exc())
    #except:
    #    print("Ha ocurrido un error: ", sys.exc_info()[0])
    finally:
        print("Websocket finalizado")

model = joblib.load("modelo_deteccion_v4.uni")

# Crear el servidor de websocket en un ip local en el puerto 5000
# De manera previa se debe apertura el puerto en el sistema operativo
start_server = websockets.serve(signal_capture, "0.0.0.0", 5000)

# Ejecutar el servidor, se ejecutar de manera continua.
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()