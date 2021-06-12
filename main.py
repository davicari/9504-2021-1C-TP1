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


def serie_fourier_altura(t, ind, amp, fases, w_0, output, save_data):
    if (save_data is True):
        output.writerow(["k", "O_k", "Q_k", "A_k", "B_k"])
    acc = 0
    for k in ind:
        a_k = ak(amp[k], fases[k])
        b_k = bk(amp[k], fases[k])

        if (save_data is True):
            output.writerow([k, amp[k], fases[k], a_k, b_k])

        acc = (a_k * np.cos(w_0 * k * t)) + (b_k * np.sin(w_0 * k * t)) + acc

    return acc


def procesar_rango(alturas, n_armonicos, save_data):
    # open files
    fft_writer = None
    sft_writer = None
    if (save_data is True):
        now = int(datetime.now().timestamp())
        fft_file = open(f'resultados/fft_csv_file_{now}.csv', 'w')
        sft_file = open(f'resultados/sft_csv_file_{now}.csv', 'w')
        fft_writer = csv.writer(fft_file)
        sft_writer = csv.writer(sft_file)

    # Parametros
    T = len(alturas)
    t = range(0, int(T))
    omega_0 = (2 * np.pi) / T

    W = int(T/2)

    # Obtencion de la transformada
    alturas_fft = np.fft.fft(alturas)
    h_alturas_fft = np.abs(alturas_fft)
    a_alturas_fft = np.angle(alturas_fft)
    h_alturas_fft_normalizadas = h_alturas_fft / W

    # Normalizamos a parte el valor medio, dado que se repite en la muestra
    h_alturas_fft_normalizadas[0] = h_alturas_fft_normalizadas[0]/2

    # Seleccion de los armonicos utilizamos solo la mitad de la muestra
    mitad = h_alturas_fft_normalizadas[0:W]
    maximos = np.flip(np.sort(mitad))[0:n_armonicos]
    h_alturas_fft_filtrados = np.where(
        mitad < np.min(maximos),
        0,
        mitad)
    indices_elementos_filtrados = np.nonzero(h_alturas_fft_filtrados)[0]

    # Guardamos los datos de la corrida.
    if (save_data is True):
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
        sft_writer,
        save_data
    )

    # Error cuadratico medio.
    ecm = np.mean((np.abs(alturas - sf_alturas)**2))
    print("ECM: ", ecm)

    # Dibujamos las alturas medidas y las alturas estimadas.
    plt.plot(t, sf_alturas, 'r-', t, alturas, 'b--')
    plt.show()


def usage():
    print('\nUso:')
    print('\tmain.py -p <P> -n <N> -s')
    print('\tmain.py --punto <P> --armonicos <N> --save')
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
    save_data = False
    try:
        opts, args = getopt.getopt(argv, "hp:n:s", ['punto=', 'armonicos=', 'save='])
        print(opts, args)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-s", "--save"):
            save_data = True
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

    print('Save data', save_data)
    ranges = read_file()
    procesar_rango(ranges[point], n_armonicos, save_data)


if __name__ == "__main__":
    main(sys.argv[1:])
