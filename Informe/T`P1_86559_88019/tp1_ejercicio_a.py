import funciones as f
import numpy as np
import pandas as pd


#El objetivo de este este ejercicio es seleccionar los armónicos que aportan información.

mediciones_alturas = f.leer_archivo_maine()['Verified (m)']
N_samples = int(len(mediciones_alturas))
tiempo = np.arange(N_samples)
mediciones_alturas_fft = f.fft_datos(mediciones_alturas)
W_samples = int(len(mediciones_alturas_fft))
omega = np.arange(W_samples) * (2*np.pi/N_samples)
freq = np.arange(W_samples) / N_samples
serie_fourier_alturas = f.sf_altura(mediciones_alturas_fft,tiempo)

#graficamos los armonicos

f.plot_log("ejercicio_a_fft_log",freq,np.abs(mediciones_alturas_fft),"Frecuencia [1/H]","Amplitud [m]")

#Comparamos el error cuadrático medio segun la cantidad de maximos de la fft que tomamos:
numero_de_armonicos = 12
pd_ecm_x_n = f.obtener_error_cuadratico_segun_numero_muestras(numero_de_armonicos,mediciones_alturas,mediciones_alturas_fft,'ecm_por_'+str(numero_de_armonicos)+'_armonicos.csv')

f.plot("ejercicio_a_ecm_x_"+str(numero_de_armonicos)+"armonicos",np.arange(len(pd_ecm_x_n["ECM%"])),pd_ecm_x_n["ECM%"],"Numero de Armonicos","Error Cuadratico Medio Porcentual")

#Se decide que con los primeros 3 armonicos se comete un error del 30%, y para bajarlo a menos de 25 se necesitan mas de 10 armónicos. 
print(pd_ecm_x_n)