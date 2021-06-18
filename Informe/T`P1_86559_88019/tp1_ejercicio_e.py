from numpy.core.numeric import indices
import funciones as f
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt


#Debemos aplicar este modelo a nuestros datos

#altura = a_0 + a_1 * cos(w_1 *t + phi_1)

#Sabemos que el valor a_0 es el valor medio
N_armonicos = 2 
mediciones_alturas_enero = f.leer_archivo_maine("2019/01/01 00:00","2019/01/31 23:00")['Verified (m)']
mediciones_alturas_marzo = f.leer_archivo_maine("2019/03/01 00:00","2019/03/31 23:00")['Verified (m)']

n_alturas_enero = len(mediciones_alturas_enero)
n_alturas_marzo = len(mediciones_alturas_marzo)

t_enero = np.arange(n_alturas_enero)
t_marzo = np.arange(n_alturas_marzo)

fft_enero = f.fft_datos(mediciones_alturas_enero)
fft_marzo = f.fft_datos(mediciones_alturas_marzo)

indices_armonicos_enero = f.obtener_indices_armonicos(fft_enero,N_armonicos)
indices_armonicos_marzo = f.obtener_indices_armonicos(fft_marzo,N_armonicos)

n_fft_enero = len(fft_enero)
n_fft_marzo = len(fft_marzo)

#ordeno los parámetros en un array [a_0,a_1,w_1,phi_1]
print("Indices Armonicos Enero",indices_armonicos_enero)
print("Indices Armonicos Marzo",indices_armonicos_marzo)

parametros_enero = [np.abs(fft_enero[0]),np.abs(fft_enero[indices_armonicos_enero[1]]), indices_armonicos_enero[1] * (2*np.pi/n_alturas_enero),np.angle(fft_enero[indices_armonicos_enero[1]])]
parametros_marzo = [np.abs(fft_marzo[0]),np.abs(fft_marzo[indices_armonicos_marzo[1]]), indices_armonicos_marzo[1] * (2*np.pi/n_alturas_marzo),np.angle(fft_marzo[indices_armonicos_marzo[1]])] 

print("Valor Medio Enero (a_0): ",parametros_enero[0],", Primer Armónico (a_1): ",parametros_enero[1]," Frecuencia Angular (w_1): ",parametros_enero[2], " Fase 1 (phi_1): ",parametros_enero[3])
print("Valor Medio marzo (a_0): ",parametros_marzo[0],", Primer Armónico (a_1): ",parametros_marzo[1]," Frecuencia Angular (w_1): ",parametros_marzo[2], " Fase 1 (phi_1): ",parametros_marzo[3])

sf_enero = f.sf_altura(fft_enero,np.arange(n_alturas_enero),indices_armonicos_enero)
sf_marzo = f.sf_altura(fft_marzo,np.arange(n_alturas_marzo),indices_armonicos_marzo)

aprox_enero = f.alturas_1(parametros_enero,t_enero)
aprox_marzo = f.alturas_1(parametros_marzo,t_marzo)

print("diff SF vs aprox Enero", f.ECM(aprox_enero,sf_enero),f.ECM(aprox_enero,mediciones_alturas_enero))
print("diff SF vs aprox Marzo", f.ECM(aprox_marzo,sf_marzo),f.ECM(aprox_marzo,mediciones_alturas_marzo))

ECM_enero = f.ECM(sf_enero,mediciones_alturas_enero)
ECM_marzo = f.ECM(sf_marzo,mediciones_alturas_marzo)

print("ECM Enero",ECM_enero)
print("ECM Marzo",ECM_marzo)

ECM_enero_marzo = f.ECM(sf_enero,mediciones_alturas_marzo)
ECM_marzo_enero = f.ECM(sf_marzo,mediciones_alturas_enero)

print("ECM Enero predicho con Marzo",ECM_enero)
print("ECM Marzo predicho con Enero",ECM_marzo)


"""
plt.plot(np.arange(n_fft_enero),np.abs(fft_enero),'b-',np.arange(n_fft_marzo),np.abs(fft_marzo),'r')
plt.yscale("log")
plt.show()
"""
plt.plot(t_enero,aprox_enero,'b',t_marzo,aprox_marzo,'r')
plt.yscale("linear")
plt.savefig(f.root_file+'/Images/aprox_enero_aprox_marzo.png', dpi=300, bbox_inches='tight')





