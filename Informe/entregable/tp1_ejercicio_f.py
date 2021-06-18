from numpy.core.defchararray import index
from numpy.core.numeric import indices
import funciones as f
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt

#En este ejercicio, las muestras del archivo Mar-del-plata.csv no son equiespaciadas y registran solo pleamares y bajamares. Y los registros estan en minutos. 
#La menor unidad de tiempo que usaremos será el minuto
#Primero procesamos el archivo y obtenemos una versión que ademas de tener las columnas de fecha y altura, se le agregan las columnas que guarden:
#t_minutos: los minutos desde la primer muestra
#i_minutos: los munutos desde la muestra anterior
#es_pleamar: si es pleamar o bajamar

f.procesar_archivo_mar_del_plata()
alturas = f.procesar_archivo_mar_del_plata_normalizado()

#luego podemos obtener, nuestro pares {x,y}

#nuevamente queremos utilizar la funcion a_0 + a_1 cos(w_1 *t + phi_1) para aproximar estos datos.
#esta funcion se puede reescribir como  a_0 * 1 + a_1 * cos(w_1 *t ) + b_1 * sen (w_1 * t)
#en este caso, el conjunto de phi_i(x) = {1,cos(w_1*t),sen(w_1*t)} y queremos saber los c_i {a_0,a_1,b_1}
#asumimos que w_1 es correspondiente a 2pi/Tpleamares, tambien podemos usar como dato que la pleamar ocurre cada 12 horas, entonces w_1 = 2pi / (12*60)

X = alturas['i_minutos']
Y = alturas['altura']

diferencias_pleamares =[]
filter = alturas["es_pleamar"] == True
tiempos_pleamares = alturas.where(filter,inplace=True)["t_minutos"]

for pleamar in tiempos_pleamares:
    print(pleamar)
    if(pleamar.index == 0):
        diferencias_pleamares.append(0)
    
    
        


"""
print(np.mean(alturas["altura"]))

print(diffs_pleamares)
T_mean = np.mean(diffs_pleamares)
print(T_mean)
w_1 = 2*np.pi / T_mean

print(w_1)
print(np.mean(Y))

c_i = f.calcular_coeficientes_c_i(X,Y,w_1)

print(c_i)
modelo = f.alturas_1_f(c_i,w_1,Y)

ECM_modelo = f.ECM(X,modelo)

print(ECM_modelo)

#print(alturas)
"""