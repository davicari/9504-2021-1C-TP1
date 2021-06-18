import funciones as f
import numpy as np
import pandas as pd
import datetime as dt

#El objetivo de este ejercicio es analizar sub muestras y procesarlas según un rango de fecha.
#El formato de las fechas es 
format = "%Y/%m/%d %H:%M"
fechas = [
    {"Descripcion":"Primer Semana de Enero" ,"sFechaInicio":"2019/01/01 00:00","sFechaFin":"2019/01/07 23:00"},
    {"Descripcion":"Segunda Semana de Enero" ,"sFechaInicio":"2019/01/08 00:00","sFechaFin":"2019/01/14 23:00"},
    {"Descripcion":"Enero y Febrero" ,"sFechaInicio":"2019/01/01 00:00","sFechaFin":"2019/03/01 00:00"},
    {"Descripcion":"Marzo y Abril" ,"sFechaInicio":"2019/01/01 00:00","sFechaFin":"2019/03/01 00:00"},
]



N_armonicos = 4
for fecha in fechas:
    print(fecha)
    mediciones_alturas = f.leer_archivo_maine(fecha["sFechaInicio"],fecha["sFechaFin"])['Verified (m)']
    N_samples = int(len(mediciones_alturas))
    tiempo = np.arange(N_samples)
    mediciones_alturas_fft = f.fft_datos(mediciones_alturas)
    W_samples = int(len(mediciones_alturas_fft))
    omega = np.arange(W_samples) * (2*np.pi/N_samples)
    freq = np.arange(W_samples) / N_samples
    #Calculo para los  4 armónicos principales
    indices_armonicos = f.obtener_indices_armonicos(mediciones_alturas_fft,N_armonicos)
    serie_fourier_alturas = f.sf_altura(mediciones_alturas_fft,tiempo,indices_armonicos)
    ecm_n = f.ECM(serie_fourier_alturas,mediciones_alturas)
    print("Las frecuencias utilizadas son :",freq[indices_armonicos])
    print("El E.C.M para el rango de fechas de "+fecha["sFechaInicio"]+" hasta "+fecha["sFechaFin"]+" con "+str(N_armonicos)+" armónicos es: ",ecm_n)
    #Calculo para los 3 armónicos principales
    indices_armonicos = f.obtener_indices_armonicos(mediciones_alturas_fft,N_armonicos-1)
    serie_fourier_alturas = f.sf_altura(mediciones_alturas_fft,tiempo,indices_armonicos)
    ecm_n = f.ECM(serie_fourier_alturas,mediciones_alturas)
    print("El E.C.M para el rango de fechas de "+fecha["sFechaInicio"]+" hasta "+fecha["sFechaFin"]+" con "+str(N_armonicos-1)+" armónicos es: ",ecm_n)
    


