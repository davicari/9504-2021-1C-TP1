import sys
import getopt
import numpy as np
import pandas
from matplotlib import pyplot as plt
from datetime import datetime
import csv


def read_file():
    mareas = pandas.read_csv('CO-OPS_8410140_met-2019.csv')
    alturas_totales = mareas['Verified (m)']

    alturas_semana_1_enero = alturas_totales[0:168]
    alturas_semana_2_enero = alturas_totales[169:336]
    alturas_ene_feb = alturas_totales[0:1416]
    alturas_mar_abr = alturas_totales[1417:2880]
    return [
        alturas_totales,
        alturas_semana_1_enero,
        alturas_semana_2_enero,
        alturas_ene_feb,
        alturas_mar_abr
    ]


def store_fft_data(indices, absolutes, angles, output):
    output.writerow(["Indice", "Abs", "Angle"])
    for i in indices:
        output.writerow([i, absolutes[i], angles[i]])


def ak(a_k, f_k):
    return a_k * np.cos(f_k)


def bk(b_k, f_k):
    return -b_k * np.sin(f_k)


def serie_fourier_altura(t, indices, amplitudes, fases, w_0, output):
    output.writerow(["k", "O_k", "Q_k", "A_k", "B_k"])
    acc = 0
    for k in indices:
        a_k = ak(amplitudes[k], fases[k])
        b_k = bk(amplitudes[k], fases[k])
        output.writerow([k, amplitudes[k], fases[k], a_k, b_k])

        acc = (a_k * np.cos(w_0 * k * t)) + (b_k * np.sin(w_0 * k * t)) + acc

    return acc


def procesar_rango(alturas, n_armonicos):
    # open files
    now = int(datetime.now().timestamp())
    fft_file = open(f'resultados/fft_csv_file_{now}.csv', 'w')
    sft_file = open(f'resultados/sft_csv_file_{now}.csv', 'w')
    fft_writer = csv.writer(fft_file)
    sft_writer = csv.writer(sft_file)

    # Parametros
    T = len(alturas)
    t = range(0, int(T))
    omega_0 = (2 * np.pi) / T

    # Obtencion de la transformada
    alturas_fft = np.fft.fft(alturas)
    h_alturas_fft = np.abs(alturas_fft)
    a_alturas_fft = np.angle(alturas_fft)
    h_alturas_fft_normalizadas = (h_alturas_fft) / T

    # Seleccion de los armonicos
    maximos = np.flip(np.sort(h_alturas_fft_normalizadas))[0:n_armonicos]
    h_alturas_fft_filtrados = np.where(
        h_alturas_fft_normalizadas < np.min(maximos),
        0,
        h_alturas_fft_normalizadas)
    indices_elementos_filtrados = np.nonzero(h_alturas_fft_filtrados)[0]

    # Guardamos los datos de la corrida.
    store_fft_data(
        indices_elementos_filtrados,
        h_alturas_fft,
        a_alturas_fft,
        fft_writer
    )

    # Calculo de las alturas.
    sf_alturas = serie_fourier_altura(
        t,
        indices_elementos_filtrados,
        h_alturas_fft_normalizadas,
        a_alturas_fft,
        omega_0,
        sft_writer
    )

    # Error cuadratico medio.
    ecm = np.mean((np.abs(alturas - sf_alturas)**2))
    print("ECM: ", ecm)

    # Dibujamos las alturas medidas y las alturas estimadas.
    plt.plot(t, sf_alturas, 'r-', t, alturas, 'b--')
    plt.show()


def usage():
    print('\nUso:')
    print('\tmain.py -p <P> -n <N>')
    print('\tmain.py --punto <P> --armonicos <N>')
    print('\nN = cantidad de armonicos (default: 15)')
    print('P = punto del tp (default: el set completo)\n')
    print('Puntos:')
    print('\t1: Set completo')
    print('\t2: Primera semana de enero')
    print('\t3: Segunda semana de enero')
    print('\t4: Enero y febrero')
    print('\t5: Marzo y abril')


def main(argv):
    point = 0
    n_armonicos = 15
    try:
        opts, args = getopt.getopt(argv, "hp:n:", ['punto=', 'armonicos='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-p", "--punto"):
            try:
                value = int(arg)
                if ((value < 1) or (value > 5)):
                    raise ValueError
                else:
                    point = value - 1
            except ValueError:
                print('-p tiene que ser un numero entre 1 y 5')
                usage()
                sys.exit(2)
        elif opt in ("-n", "--armonicos"):
            try:
                n_armonicos = int(arg)
            except ValueError:
                print('-n tiene que ser un numero entero')
                usage()
                sys.exit(2)

    ranges = read_file()
    procesar_rango(ranges[point], n_armonicos)


if __name__ == "__main__":
    main(sys.argv[1:])
