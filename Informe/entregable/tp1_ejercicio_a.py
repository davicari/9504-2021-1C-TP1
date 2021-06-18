import funciones as f
import numpy as np


#El objetivo de este este ejercicio es obtener una descomposición armónica mediante la FFT

mediciones_alturas = f.leer_archivo_maine()['Verified (m)']

N_samples = int(len(mediciones_alturas))
tiempo = np.arange(N_samples)
mediciones_alturas_fft = f.fft_datos(mediciones_alturas)

W_samples = int(len(mediciones_alturas_fft))
omega = np.arange(W_samples) * (2*np.pi/N_samples)

freq = np.arange(W_samples) / N_samples
serie_fourier_alturas = f.sf_altura(mediciones_alturas_fft,tiempo)

# armonicos_maximos =  f.obtener_indices_armonicos(np.abs(mediciones_alturas_fft),5)
# print(armonicos_maximos)

#f.plot_log("ejercicio_a_fft_log",freq,np.abs(mediciones_alturas_fft),"Frecuencia [1/H]","Amplitud [m]",True)

#Comparamos el error cuadrático medio segun la cantidad de maximos de la fft que tomamos:
ECM_N = []
media_ = np.mean(mediciones_alturas)
for i in np.arange(10):
    numero_armonicos = i+1
    indices_n = f.obtener_indices_armonicos(mediciones_alturas_fft,numero_armonicos)
    
    serie_fourier_alturas_h = f.sf_altura(mediciones_alturas_fft,tiempo,indices_n)
    error = f.ECM(serie_fourier_alturas_h,mediciones_alturas)
    print(media_, numero_armonicos, error)
    

