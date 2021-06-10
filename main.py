import numpy as np
import pandas
from matplotlib import pyplot as plt 



def A_k(a_k,f_k):
    return a_k*np.cos(f_k)
def B_k(a_k,f_k):
    return -a_k*np.sin(f_k)

def serie_fourier_altura(t,indices,amplitudes,fases,w_0):
    acumulador=0
    for k in indices:
        acumulador = A_k(amplitudes[k],fases[k])*np.cos(w_0*k * t ) + B_k(amplitudes[k],fases[k])*np.sin(w_0* k * t )  + acumulador
    return acumulador

hora = 1
T = 365*24*hora
deltaT = hora
criterio = 0.05

t = range(0,int(T/deltaT))
omega_0 = (2*np.pi*deltaT)/T
omega = omega_0 * np.array(t)
freq = omega * 2*np.pi


#cantidad de armonicos elegidos
n_armonicos_elegidos = 11
mareas = pandas.read_csv('CO-OPS_8410140_met-2019.csv')



alturas = mareas['Verified (m)']

alturas_promedio = alturas.mean()

alturas_fft = np.fft.fft(alturas) 
h_alturas_fft = np.abs(alturas_fft)
a_alturas_fft = np.angle(alturas_fft)
#Normalizo las alturas usando el valor medio
#h_alturas_fft_normalizadas = h_alturas_fft * (alturas_promedio/h_alturas_fft[0])
h_alturas_fft_normalizadas = h_alturas_fft  * deltaT/T
h_alturas_fft_media = np.mean(h_alturas_fft_normalizadas)
h_alturas_fft_promedio = sum(abs(h_alturas_fft_normalizadas[1:]))/len(h_alturas_fft_normalizadas[1:])



filtro = criterio*np.amax(h_alturas_fft_normalizadas)# Este criterio es malo, prefiero elegir por el numero de armónicos mas altos. los 5 o 6 mas altos.
print(criterio,filtro)
# filtro = np.partition(h_alturas_fft_normalizadas.flatten(),-n_armonicos_elegidos)[n_armonicos_elegidos]
# print(filtro)
# print('test')

maximos = np.flip(np.sort(h_alturas_fft_normalizadas))[0:n_armonicos_elegidos]

print("max",maximos)


h_alturas_fft_filtrados = np.where(h_alturas_fft_normalizadas <  np.min(maximos), 0,h_alturas_fft_normalizadas)


indices_elementos_filtrados = np.nonzero(h_alturas_fft_filtrados)[0]
print(len(indices_elementos_filtrados),indices_elementos_filtrados)
sf_alturas = serie_fourier_altura(t,indices_elementos_filtrados,h_alturas_fft_normalizadas,a_alturas_fft,omega_0)


mse = np.mean((np.abs(alturas - sf_alturas)**2))
print("ECM: ",mse)
plt.plot(t,sf_alturas,'r-',t,alturas,'b--')
plt.show()

# print(filtro)
# print(omega.size)
# print(h_alturas_fft.size)
# print(h_alturas_fft_filtrado)
# print(h_alturas_fft_media,h_alturas_fft_promedio)
# plt.plot(freq,h_alturas_fft_filtrado,'r',freq,np.full((omega.size,1),h_alturas_fft_media),'b')
# plt.show()