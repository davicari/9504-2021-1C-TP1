import numpy as np
import pandas
from matplotlib import pyplot as plt 



def serie_fourier_altura(t,indices,amplitudes,fases,w_0):
    acumulador=0
    for k in indices:
        acumulador = amplitudes[k]*np.cos(w_0*k * t +fases[k]) + acumulador
    return acumulador

hora = 3600
T = 365*24*hora
deltaT = hora
criterio = 0.05

t = range(0,int(T/deltaT))
omega_0 = (2*np.pi*deltaT)/T
omega = omega_0 * np.array(t)
freq = omega * 2*np.pi

n_armonicos_elegidos = 6

mareas = pandas.read_csv('CO-OPS_8410140_met-2019.csv')



#print(data)

alturas = mareas['Verified (m)']
#chajá! calcular el valor medio sacando el valor medio.

alturas_promedio = alturas.mean()
alturas_fft = np.fft.fft(alturas) 
h_alturas_fft = np.abs(alturas_fft)
a_alturas_fft = np.angle(alturas_fft)
#Normalizo las alturas usando el valor medio
h_alturas_fft_normalizadas = h_alturas_fft * (alturas_promedio/h_alturas_fft[0])
h_alturas_fft_media = np.mean(h_alturas_fft_normalizadas)
h_alturas_fft_promedio = np.average(h_alturas_fft_normalizadas)


#filtro = criterio*np.max(h_alturas_fft) Este criterio es malo, prefiero elegir por el numero de armónicos mas altos. los 5 o 6 mas altos.
filtro = np.partition(h_alturas_fft_normalizadas.flatten(),-n_armonicos_elegidos)[-n_armonicos_elegidos]

h_alturas_fft_filtrado = np.where(h_alturas_fft_normalizadas <  filtro, 0,h_alturas_fft_normalizadas)


indices_elementos_filtrados = np.nonzero(h_alturas_fft_filtrado)[0]

print(indices_elementos_filtrados)

sf_alturas = serie_fourier_altura(t,indices_elementos_filtrados,h_alturas_fft_normalizadas,a_alturas_fft,omega_0)



plt.plot(t,sf_alturas,'r^',t,alturas,'b-')
plt.show()



# print(filtro)
# print(omega.size)
# print(h_alturas_fft.size)
# print(h_alturas_fft_filtrado)
# print(h_alturas_fft_media,h_alturas_fft_promedio)
# plt.plot(freq,h_alturas_fft_filtrado,'r',freq,np.full((omega.size,1),h_alturas_fft_media),'b')
# plt.show()