import numpy as np
import pandas
from matplotlib import pyplot as plt

mareas = pandas.read_csv('Mar-del-plata.csv')
alturas = mareas['altura']

T = len(alturas)
x = range(0, int(T))
w_0 = 2*np.pi/T
a_0 = np.mean(alturas)
res = 0
medidos = []


def q1(t):
    return np.cos(w_0 * t)


def q2(t):
    return np.sin(w_0 * t)


Q1 = sum(q1(idx)*q1(idx) for idx, val in enumerate(alturas))/len(alturas)
Q2 = sum(q1(idx)*q2(idx) for idx, val in enumerate(alturas))/len(alturas)
Q3 = sum(q2(idx)*q1(idx) for idx, val in enumerate(alturas))/len(alturas)
Q4 = sum(q2(idx)*q2(idx) for idx, val in enumerate(alturas))/len(alturas)

Y1 = sum(value*q1(idx) for idx, value in enumerate(alturas))/len(alturas)
Y2 = sum(value*q2(idx) for idx, value in enumerate(alturas))/len(alturas)


a = np.matrix([[Q1, Q2], [Q3, Q4]])
b = np.array([Y1, Y2])
c = np.linalg.solve(a, b)

c1 = c[0]
c2 = c[1]

print(f'a_0 = {a_0}, T = {T}, c1 = {c1}, c2 = {c2}')


def f(x):
    return a_0 + (c1 * np.cos(w_0 * x)) + (c2 * np.sin(w_0 * x))


for i in x:
    y_medido = f(i)
    medidos.append(y_medido)


plt.plot(x, alturas, 'r-', x, medidos, 'b--')
plt.show()

ecm = np.mean(np.abs(medidos - alturas)**2)
print('ECM', ecm)
