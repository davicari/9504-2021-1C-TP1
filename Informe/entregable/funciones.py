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

def procesar_archivo_mar_del_plata():
    #leo el archivo
    alturas_mareas = pd.read_csv(root_file+"/Mar-del-plata.csv")
    fechas = alturas_mareas['fecha_hora']
    alturas = alturas_mareas['altura']
    valor_medio_marea = np.mean(alturas)
    format = "%d/%m/%y %H:%M:%S"
    diffs = []
    minutes = []
    pleamares = []
    for index, f in enumerate(fechas):
        es_pleamar = alturas[index] > valor_medio_marea
        pleamares.append(es_pleamar)
        if (index == 0):
            diffs.append(0.0)
            minutes.append(0.0)
        else:
            d0 = datetime.strptime(fechas[index - 1], format)
            d1 = datetime.strptime(f, format)
            secs = (d1 - d0).total_seconds()/60
            next_val = secs + minutes[index - 1]
            minutes.append(int(next_val))
            diffs.append(int(secs))

    alturas_mareas["t_minutos"] = minutes
    alturas_mareas["i_minutos"] = diffs
    alturas_mareas["es_pleamar"] = pleamares

    alturas_mareas.to_csv(root_file+"/Mar-Del-Plata-Normalizado.csv")

def procesar_archivo_mar_del_plata_normalizado():
    return pd.read_csv(root_file+"/Mar-Del-Plata-Normalizado.csv")

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
def ECM(A,B):
    return np.sqrt(np.mean((A - B)**2))

#Esta funcion tabula el error cuadrático medio, segun la cantidad de armónicos utilizados para calcular la serie de fourier
def obtener_error_cuadratico_segun_numero_muestras(n,mediciones,mediciones_fft,filename=""):
    data = []
    media_ = np.mean(mediciones)
    for i in np.arange(n):
        numero_armonico = i+1
        indices_n = obtener_indices_armonicos(mediciones_fft,numero_armonico)
        serie_fourier_alturas_h = sf_altura(mediciones_fft,np.arange(len(mediciones)),indices_n)
        ecm = ECM(mediciones,serie_fourier_alturas_h)
        #print(numero_armonico, ecm, (ecm/media_ * 100))
        data.append([numero_armonico,ecm,(ecm/media_ * 100)])
    ec_panda = pd.DataFrame(data,columns=["Numero de Armónico","E.C.M.","ECM%"])
    if(filename != ""):
        ec_panda.to_csv(filename)
    return ec_panda


#en el informe se muestra como llegamos a estas ecuaciones
def calcular_coeficientes_c_i(minutes,alturas,w_1):
    Q11 = sum(1 for i,y in enumerate(alturas) )
    Q22 = sum(np.cos(w_1*minutes[i])**2 for i, x in enumerate(alturas))
    Q33 = sum(np.sin(w_1*minutes[i])**2 for i, x in enumerate(alturas))
    Q12 = sum(np.cos(w_1*minutes[i]) for i, x in enumerate(alturas))
    Q23 = sum(np.sin(w_1*minutes[i]) for i, x in enumerate(alturas))
    Q13 = sum(np.cos(w_1*minutes[i])*np.sin(w_1*minutes[i]) for i, x in enumerate(alturas))

    Y1 = sum(y for i, y in enumerate(alturas))
    Y2 = sum(y*np.cos(w_1*minutes[i]) for i, y in enumerate(alturas))
    Y3 = sum(y*np.sin(w_1*minutes[i]) for i, y in enumerate(alturas))

    
    a = np.matrix([[Q11, Q12, Q13],[Q12,Q22,Q23],[Q13,Q23,Q33]])
    b = np.array([Y1, Y2, Y3])
    c = np.linalg.solve(a, b)
    return c


#Esta funcion representa a la funcion alturas para el primer armónico.
#params[0] = a_0
#params[1] = a_1
#params[2] = w_1
#params[3] = phi_1
def alturas_1(params,t):
    return params[0] + params[1]*np.cos(params[2]*t+params[3])

def alturas_1_f(params,w_1,t):
    return params[0] + params[1]*np.cos(w_1*t)- params[2]*np.sin(w_1*t)
    

def plot_log(name,x,y,x_label,y_label,show = False):
    plt.plot(x,y,'b-')
    plt.yscale("log")
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.savefig( root_file+'/plots/'+name+'.png', dpi=300, bbox_inches='tight')
    if(show):
        plt.show()
    
def plot(name,x,y,x_label,y_label,show = False):
    plt.plot(x,y,'b-')
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.yscale("linear")
    plt.savefig( root_file+'/plots/'+name+'.png', dpi=300, bbox_inches='tight')
    if(show):
        plt.show()

