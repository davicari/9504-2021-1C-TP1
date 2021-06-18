
from numpy.core.defchararray import index
from numpy.core.numeric import indices
import funciones as f
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt

alturas = f.leer_archivo_mar_del_plata_normalizado()

minutes= alturas["t_minutos"]
#Del 21/05/2021 el ultimo dato es a las 19:49. 
#hay 11 minutos hasta las 20
#hay 251 minutos hasta las 00:00 del 22.
#hay 10 dias del 22 al primero de junio 10*24*60 +251 = 

t_inicial = (int)(minutes[len(minutes)-1] + 251+ 10*24*60)
#10 horas
t_final = t_inicial + 60

t_junio = np.arange(t_inicial,t_final,1)

calculado_junio = f.alturas_1_g(t_junio)

t_junio_maximos = np.argmax(calculado_junio)



print("El primer máximo es a los ",t_junio_maximos+1,"minutos del primero de junio")


