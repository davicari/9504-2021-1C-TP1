import funciones as f
import numpy as np
import pandas as pd


#El objetivo de este este ejercicio es obtener una descomposición armónica mediante la FFT, 
#y obtener los coeficientes de fourier a partir de las N armónicos elegidos del ejercicio anterior
#En este caso N_armonicos = 2
N_armonicos = 2
mediciones_alturas = f.leer_archivo_maine()['Verified (m)']
N_samples = int(len(mediciones_alturas))
tiempo = np.arange(N_samples)
mediciones_alturas_fft = f.fft_datos(mediciones_alturas)
W_samples = int(len(mediciones_alturas_fft))
omega = np.arange(W_samples) * (2*np.pi/N_samples)
freq = np.arange(W_samples) / N_samples
indices_armonicos = f.obtener_indices_armonicos(mediciones_alturas_fft,N_armonicos)
serie_fourier_alturas = f.sf_altura(mediciones_alturas_fft,tiempo,indices_armonicos)

ecm_n = f.ECM(serie_fourier_alturas,mediciones_alturas)

print("El E.C.M para 2 armónicos es: ",ecm_n)
