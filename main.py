import numpy as np
import pandas
from matplotlib import pyplot as plt 



hora = 3600
T = 365*24*hora
deltaT = hora
criterio = 0.05

t = range(0,int(T/deltaT))

omega_0 = (2*np.pi*deltaT)/T

omega = omega_0 * np.array(t)

freq = omega * 2*np.pi

mareas = pandas.read_csv('CO-OPS_8410140_met-2019.csv')

#print(data)

alturas = mareas['Verified (m)']

#chajá! calcular el valor medio sacando el valor medio.


alturas_fft = np.fft.fft(alturas) 

h_alturas_fft = np.abs(alturas_fft) /2

h_alturas_fft_media = np.mean(h_alturas_fft)

h_alturas_fft_promedio = np.average(h_alturas_fft)

filtro = criterio*np.max(h_alturas_fft)

h_alturas_fft_filtrado = np.where(h_alturas_fft <  filtro, 0,h_alturas_fft)

print(filtro)
print(omega.size)
print(h_alturas_fft.size)


print(h_alturas_fft_media,h_alturas_fft_promedio)

plt.plot(freq,h_alturas_fft_filtrado,'r',freq,h_alturas_fft,'g',freq,np.full((omega.size,1),h_alturas_fft_media),'b')

plt.show()