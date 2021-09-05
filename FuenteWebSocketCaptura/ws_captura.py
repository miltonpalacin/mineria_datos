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

# WS?.sendText("BEGIN,$ACTION,$SEX,$AGE,$NAME")
# {action}_{sex}_{age}_{date}.csv

# Función que procesa el mensaje del websocket
# en este caso recibe la señal de la conexión que realiza el celular
user_list = {}
async def message_process(message):

    # Variable lista para guardar los nombres de los 
    # archivos de los usuarios que se captura la señal
    global user_list
    # Variable con el nombre del usuario
    user_name = message[message.rfind(",") + 1 : ]
    
    if message.startswith("BEGIN,"):
        
        data = message.split(",")
        # Crear el archivo CSV en caso no es creado
        date = datetime.today().strftime("%Y%m%d%H%M%S")
        # {name}_{action}_{sex}_{age}_{date}.csv
        file_name = data[4].lower() +"_" + data[1].lower() +"_" + data[2].lower() +"_" + data[3].lower() +"_" + date +".csv"

        # Crear la cabecera incial del archivo
        with open(file_name,'a') as fd:
            fd.write("MM,dd,HH,mm,ss,sss,x,y,z,actividad,sexo,edad,nombre")
        
        # Guardar el nombre para su futura consulta
        user_list[user_name] = file_name

        print(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"Captura iniciada de:",user_name, ", en archivo:", file_name)

    elif message.startswith("END,"):
        # Finalizar captura
        print(datetime.today().strftime("%Y-%m-%d %H:%M:%S"),"Captura finalizada de:",user_name, ",en archivo:", user_list[user_name])
    else:
        # Guardar datos de captura de señal
        with open(user_list[user_name],'a') as fd:
            fd.write("\n")
            fd.write(message)

# Función controlador de conexiones:
# el presenta controlador solo se encargará de recibir la señal
# no responderá (enviará) mensajes de respuesta o feedback al cliente.
async def signal_capture(websocket, path):

    # Procesar los mensajes de forma continua sin cerrar 
    # la conexión con los clientes
    try:
        async for message in websocket:
            #print(message)
            await message_process(message)
    except Exception as e:
        print ("Ha ocurrido un error:", e)
        print (traceback.format_exc())
    #except:
    #    print("Ha ocurrido un error: ", sys.exc_info()[0])
    finally:
        print("Websocket finalizado")


# Crear el servidor de websocket en un ip local en el puerto 5000
# De manera previa se debe apertura el puerto en el sistema operativo
start_server = websockets.serve(signal_capture, "0.0.0.0", 5000)

# Ejecutar el servidor, se ejecutar de manera continua.
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()