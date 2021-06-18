import numpy as np
import pandas
from matplotlib import pyplot as plt
from datetime import datetime
import csv

fft_file = open('resultados/diff_minutes.csv', 'w')
fft_writer = csv.writer(fft_file)
fft_writer.writerow(['diferencia', 'acumulado','pleamar'])

mareas = pandas.read_csv('Mar-del-plata.csv')
alturas = mareas['altura']
fechas = mareas['fecha_hora']

#obtengo las alturas sin el valor de continua
alturas_0 = alturas-np.mean(alturas)
#obtengo los valores de las pleamares, o sea positivos
alturas_pleamares = np.where( alturas < 0 ,0,alturas)

#Esto es una aproximación, como diría un ex profesor "a ojo de buen cubero" de la amplitud esperable para las pleamares respecto del valor medio.
a_1 = ( np.amax(alturas_pleamares) - np.amin(alturas_pleamares)) / 2
#Este como sabemos es el valor medio, 
a_0 = np.mean(alturas)

#
#alturas_2 = np.where(alturas < a_0, 0, alturas) - a_0

diffs = []
minutes = []
format = "%d/%m/%y %H:%M:%S"
for index, f in enumerate(fechas):
    es_pleamar = alturas[index] > a_0

    if (index == 0):
        diffs.append(0.0)
        minutes.append(0.0)
    else:
        d0 = datetime.strptime(fechas[index - 1], format)
        d1 = datetime.strptime(f, format)
        secs = (d1 - d0).total_seconds()/60
        next_val = secs + minutes[index - 1]
        minutes.append(float(next_val))
        diffs.append(float(secs))
        
        fft_writer.writerow([float(secs), float(next_val),bool(es_pleamar)],)


fft_file.close()
# a = np.matrix([[Q1, Q2], [Q3, Q4]])
# b = np.array([Y1, Y2])
# c = np.linalg.solve(a, b)
#
# c1 = c[0]
# c2 = c[1]

# print(f'a_0 = {a_0}, T = {T}, c1 = {c1}, c2 = {c2}')


# def f(x):
#     return (c1 * np.cos(w_0 * x)) + (c2 * np.sin(w_0 * x))


# M * C = Y
#E

mareas_normalizadas = pandas.read_csv("resultados/diff_minutes.csv")
filter = mareas_normalizadas["pleamar"] == True
mareas_normalizadas.where(filter,inplace=True)
diffs_pleamares = np.mean(mareas_normalizadas['acumulado'])/len(mareas_normalizadas)
T_mean = np.mean(diffs_pleamares)

Y = alturas
om = 2*np.pi / T_mean
X = minutes


def f_1(x):
    return np.cos(C*x)


def f_2(x):
    return np.sin(C*x)

"""
Q1 = sum(f_1(diffs[i])*f_1(diffs[i]) for i, x in enumerate(alturas))
Q2 = sum(f_1(diffs[i])*f_2(diffs[i]) for i, x in enumerate(alturas))
Q3 = sum(f_2(diffs[i])*f_1(diffs[i]) for i, x in enumerate(alturas))
Q4 = sum(f_2(diffs[i])*f_2(diffs[i]) for i, x in enumerate(alturas))
Y1 = sum(alturas[i]*f_1(diffs[i]) for i, x in enumerate(alturas))
Y2 = sum(alturas[i]*f_2(diffs[i]) for i, x in enumerate(alturas))
"""


Q11 = sum(1 for i,y in enumerate(alturas) )
Q22 = sum(np.cos(om*minutes[i])**2 for i, x in enumerate(alturas))
Q33 = sum(np.sin(om*minutes[i])**2 for i, x in enumerate(alturas))
Q12 = sum(np.cos(om*minutes[i]) for i, x in enumerate(alturas))
Q23 = sum(np.sin(om*minutes[i]) for i, x in enumerate(alturas))
Q13 = sum(np.cos(om*minutes[i])*np.sin(om*minutes[i]) for i, x in enumerate(alturas))

Y1 = sum(y for i, y in enumerate(alturas))
Y2 = sum(y*np.cos(om*minutes[i]) for i, y in enumerate(alturas))
Y3 = sum(y*np.sin(om*minutes[i]) for i, y in enumerate(alturas))

print( Q11,Q22,Q33,Q12,Q13,Q23)

a = np.matrix([[Q11, Q12, Q13],[Q12,Q22,Q23],[Q13,Q23,Q33]])
b = np.array([Y1, Y2, Y3])

c = np.linalg.solve(a, b)

print("Constantes ",c)

# c1 =  B * cos(D) => D = fi_1; B = a_1
# c2 = -B * sin(D) => D = fi_1; B = a_1
def g(x):
    return c[0]*1 + c[1]*np.cos(om*x)+c[2]*np.sin(om*x)


calculados = []
for min in minutes:
    val = g(min)
    calculados.append(val)


ecm = np.mean(np.abs(calculados - alturas)**2)
print(f'Media utilizada: {a_0}')
print('ECM', ecm)


t = range(0, len(alturas))
plt.plot(t, alturas, 'r-', t, calculados, 'g--')
plt.show()

#Del 21/05/2021 el ultimo dato es a las 19:49. 
#hay 11 minutos hasta las 20
#hay 251 minutos hasta las 00:00 del 22.
#hay 10 dias del 22 al primero de junio 10*24*60 +251 = 

t_inicial = (int)(minutes[len(minutes)-1] + 251+ 10*24*60)
#10 horas
t_final = t_inicial + 60

t_junio = range(t_inicial,t_final,1)
calculado_junio = g(t_junio)

t_junio_maximos = np.argmax(calculado_junio)

print("El primer máximo es a los ",t_junio_maximos+1,"minutos del primero de junio")



