import numpy as np
import pandas
from matplotlib import pyplot as plt
from datetime import datetime
import csv

fft_file = open('resultados/diff_minutes.csv', 'w')
fft_writer = csv.writer(fft_file)
fft_writer.writerow(['diferencia', 'acumulado'])

mareas = pandas.read_csv('Mar-del-plata.csv')
alturas = mareas['altura']
fechas = mareas['fecha_hora']
a_0 = np.mean(alturas)

print(np.amax(alturas))
alturas_2 = np.where(alturas < a_0, 0, alturas) - a_0

diffs = []
minutes = []
format = "%d/%m/%y %H:%M:%S"
for index, f in enumerate(fechas):
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
        fft_writer.writerow([float(secs), float(next_val)])

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
Y = alturas
C = int(np.mean(diffs))
X = minutes

print(f'alturas: {len(alturas_2)}; minutes: {len(minutes)}')


def f_1(x):
    return np.cos(C*x)


def f_2(x):
    return np.sin(C*x)


Q1 = sum(f_1(diffs[i])*f_1(diffs[i]) for i, x in enumerate(alturas))
Q2 = sum(f_1(diffs[i])*f_2(diffs[i]) for i, x in enumerate(alturas))
Q3 = sum(f_2(diffs[i])*f_1(diffs[i]) for i, x in enumerate(alturas))
Q4 = sum(f_2(diffs[i])*f_2(diffs[i]) for i, x in enumerate(alturas))

Y1 = sum(alturas_2[i]*f_1(diffs[i]) for i, x in enumerate(alturas))
Y2 = sum(alturas_2[i]*f_2(diffs[i]) for i, x in enumerate(alturas))

a = np.matrix([[Q1, Q2], [Q3, Q4]])
b = np.array([Y1, Y2])

c = np.linalg.solve(a, b)

print(c)


# c1 =  B * cos(D) => D = fi_1; B = a_1
# c2 = -B * sin(D) => D = fi_1; B = a_1
def g(x):
    return a_0 + c[0] * np.cos(C*x) + c[1] * np.sin(C*x)


medidos = []
for min in minutes:
    val = g(min)
    medidos.append(val)


t = range(0, len(alturas))
plt.plot(t, alturas_2, 'r-', t, medidos, 'g--')
plt.show()

# ecm = np.mean(np.abs(medidos - alturas)**2)
# print('ECM', ecm)
