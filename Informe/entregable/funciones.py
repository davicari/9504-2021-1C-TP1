import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

root_file = os.path.dirname(os.path.realpath(__file__))

#Esta funcion obtiene como entrada dos fechas en formato "AAAA/MM/DD HH:mm". En caso de invocarse vacía devuelve todo el set.
def leer_archivo_maine(fecha_inicio = "",fecha_fin = ""):
    #Defino un Formato de Fecha
    format = "%Y/%m/%d %H:%M"
    #Cargo un objeto del tipo Panda, con el resultado del archivo
    alturas_mareas = pd.read_csv('CO-OPS_8410140_met-2019.csv', 
        parse_dates = {'DateTime': [0,1]},
        date_parser = lambda x: datetime.strptime(x,format)
    )
    #Pregunto si hay parámetros de entrada.
    if( fecha_inicio == "" or fecha_fin == ""):
        return alturas_mareas
    else:
        #Seteo una mascara con condicion de verdad para los resultados que busco
        mask = ((alturas_mareas["DateTime"] <= datetime.strptime(fecha_fin,format)) & (alturas_mareas["DateTime"] >= datetime.strptime(fecha_inicio,format)))
        return alturas_mareas.loc[mask]


def fft_datos(datos):
    #Obtengo el numero de datos
    N = len(datos)
    #Normalizo los Datos, dividiendolos por la cantidad de muestras útiles
    datos_fft = np.fft.fft(datos)[:int(N/2)]*2/N
    #Debido al shifting, el armónico 0 se duplica, dado que el espectro se repite a partir de la posicion N+1, que se vuelve a copiar en la posicion 0
    datos_fft[0] = datos_fft[0]/2
    return datos_fft
    


#Defino el coeficiente A_k según lo calculado
def A_k(a_k, f_k):
    return a_k * np.cos(f_k)
#Defino el coeficiente B_k según lo calculado
def B_k(b_k, f_k):
    return -b_k * np.sin(f_k)

#Utilizo la serie de fourier para reconstruir la señal
#datos_fft son todos los coeficientes de fourier complejos
#ind corresponde a los indices de los armónicos deseados
#t corresponde al array de tiempo. Este debe estar en la misma unidad que el tiempo de las muestras a partir se calculo la fft.
def sf_altura(datos_fft,t,ind = []):

    w_0 = 2*np.pi/(2*len(datos_fft))
    acc = 0
    #si no tengo armonicos elegidos, itero sobre todos los datos
    if(len(ind)==0):
        ind = np.arange(len(datos_fft))
    #itero sobre los armonicos seleccionados
    for k in ind:
        amp = np.abs(datos_fft[k])
        ang = np.angle(datos_fft[k])
        a_k = A_k(amp,ang)
        b_k = B_k(amp,ang)
        acc = (a_k * np.cos(w_0 * k * t)) + (b_k * np.sin(w_0 * k * t)) + acc
    return acc

#Esta funcion obtiene los indices de los armónicos mas altos
def obtener_indices_armonicos(datos_fft,n_armonicos):
    #Primero Ordeno los elementos de menor a mayor
    #Luego invierto el array para que quede de mayor a menor
    #Por ultimo tomo los n_armonicos mas altos
    maximos = np.flip(np.sort(datos_fft))[0:n_armonicos]
    #Luego filtro a los elementos menores al menor de los maximos, los vuelvo 0
    datos_fft_mayores_a_maximos = np.where(
        datos_fft < np.min(maximos),
        0,
        datos_fft)
    #Regreso los elementos 
    return np.nonzero(datos_fft_mayores_a_maximos)[0]

#implemento la definicion del Error Cuadratico Medio
def ECM(funcion_1,funcion_2):
    return np.mean(np.square(np.subtract(funcion_1,funcion_2)**2))



def plot_log(name,x,y,x_label,y_label,show = False):
    plt.plot(x,y,'b-')
    plt.yscale("log")
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.legend(loc='best')
    plt.savefig( root_file+'/plots/'+name+'.png', dpi=300, bbox_inches='tight')
    if(show):
        plt.show()
    
def plot(name,x,y,x_label,y_label,show = False):
    plt.plot(x,y,'b-')
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.legend(loc='best')
    plt.savefig( root_file+'/plots/'+name+'.png', dpi=300, bbox_inches='tight')
    if(show):
        plt.show()

