# Projeto 8 - Parte 1: exercícios 1 a 7 (AM)
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from suaBibSignal import signalMeu
import projeto8_parte1 as p8

fs = 44100
sinal = signalMeu()


def dois_maiores_picos(x):
    xf, yf = sinal.calcFFT(x, fs)
    idx = np.argsort(yf)[::-1]
    escolhidos = []
    for i in idx:
        if all(abs(xf[i] - xf[j]) > 200 for j in escolhidos):
            escolhidos.append(i)
        if len(escolhidos) == 2:
            break
    return sorted(xf[escolhidos])


def ex1():
    print("\n=== Exercício 1: SOMAR informação (2 kHz) + portadora (15 kHz) ===")
    t = np.arange(0, 2, 1 / fs)
    m = np.sin(2 * np.pi * 2000 * t)
    c = np.sin(2 * np.pi * 15000 * t)
    s = m + c
    print("Picos da soma (Hz):", np.round(dois_maiores_picos(s)))
    fig, ax = plt.subplots(figsize=(9, 4))
    p8.plot_fft(ax, s, fs, "Ex.1 - FFT de m(t) + C(t): componente em 2 kHz FORA da banda 12–18 kHz",
                (12000, 18000), 20000)
    fig.tight_layout(); fig.savefig("ex1_soma.png", dpi=120)


def ex2():
    print("\n=== Exercício 2 ===")
    print("S(t) tem componentes em fc-fm = 13 kHz e fc+fm = 17 kHz -> banda mínima: 13 kHz a 17 kHz (4 kHz = 2·fm)")


def ex3():
    print("\n=== Exercício 3: MULTIPLICAR ===")
    t = np.arange(0, 2, 1 / fs)
    m = np.sin(2 * np.pi * 2000 * t)
    c = np.sin(2 * np.pi * 15000 * t)
    s = m * c
    print("Picos da multiplicação (Hz):", np.round(dois_maiores_picos(s)), "-> esperado 13000 e 17000")
    fig, ax = plt.subplots(figsize=(9, 4))
    p8.plot_fft(ax, s, fs, "Ex.3 - FFT de m(t)·C(t): picos em 13 kHz e 17 kHz (dentro da banda 12–18 kHz)",
                (13000, 17000), 20000)
    ax.axvspan(12, 18, color="green", alpha=0.08)
    fig.tight_layout(); fig.savefig("ex3_multiplicacao.png", dpi=120)


def ex4():
    print("\n=== Exercício 4: m(t) com 20 Hz a 2000 Hz, portadora 15 kHz, banda 14–16 kHz ===")
    t = np.arange(0, 2, 1 / fs)
    m = sum(np.sin(2 * np.pi * f * t) for f in np.linspace(20, 2000, 40))
    m = m / np.max(np.abs(m))
    c = np.sin(2 * np.pi * 15000 * t)
    s_sem = m * c
    m_filt = p8.passa_baixa(m, 1000, fs)          # fm máx = 1 kHz, pois 15k ± 1k = 14k a 16k
    s_com = m_filt * c
    print("Energia fora de 14–16 kHz SEM filtrar : {:.2%}".format(p8.fracao_fora(s_sem, fs, 14000, 16000)))
    print("Energia fora de 14–16 kHz COM passa-baixa 1 kHz: {:.4%}".format(p8.fracao_fora(s_com, fs, 14000, 16000)))
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    p8.plot_fft(axs[0], s_sem, fs, "Ex.4 - modulado SEM filtrar m(t): 13–17 kHz (invade fora de 14–16)", (14000, 16000), 20000)
    p8.plot_fft(axs[1], s_com, fs, "Ex.4 - modulado COM passa-baixa de 1 kHz em m(t): 14–16 kHz", (14000, 16000), 20000)
    fig.tight_layout(); fig.savefig("ex4_filtro_antes.png", dpi=120)


def ex5():
    print("\n=== Exercício 5: áudio, portadora 16 kHz, banda 14–18 kHz ===")
    import os
    arq = "audio1.wav"
    if not os.path.exists(arq):
        p8.gerar_audios_demo()
    f, x = p8.carregar_audio(arq)
    fs_ = f
    FC, FM = 16000, 1900                                 # banda 16k ± 1,9k cabe em 14–18 kHz
    tocar_audio = "--sem-audio" not in sys.argv

    p8.tocar(x, fs_, "1) áudio original", tocar_audio)
    m = p8.preparar_mensagem(x, fs_, FM)                 # 3) tratamento: passa-baixa em 1,9 kHz e sem DC
    c = p8.portadora(FC, len(m), fs_)                    # 2) portadora com o mesmo nº de pontos
    s = m * c                                            # 3) modulação
    print("Energia fora de 14–18 kHz no modulado: {:.4%}".format(p8.fracao_fora(s, fs_, 14000, 18000)))
    p8.tocar(s, fs_, "5) sinal modulado", tocar_audio)
    d = p8.demodular(s, FC, fs_, FM)                     # 6) demodulação + passa-baixa
    print("Correlação demodulado x mensagem tratada: {:.4f}".format(p8.correlacao(d, m)))
    p8.tocar(d, fs_, "8) sinal demodulado", tocar_audio)

    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    p8.plot_fft(axs[0], x, fs_, "Ex.5 (1) FFT do áudio original", fmax=6000)
    p8.plot_fft(axs[1], s, fs_, "Ex.5 (4) FFT do sinal modulado (16 kHz ± 1,9 kHz)", (14000, 18000), 20000)
    p8.plot_fft(axs[2], d, fs_, "Ex.5 (7) FFT do sinal demodulado", fmax=6000)
    fig.tight_layout(); fig.savefig("ex5_am_audio.png", dpi=120)


def ex6_7():
    print("\n=== Exercício 6 (64-QAM) ===")
    bits, f, per = 6, 2.4e9, 2
    simb = f / per
    print("{} bits/símbolo; 1 símbolo a cada {} períodos -> {:.2e} símbolos/s -> {:.1f} Gbit/s".format(
        bits, per, simb, simb * bits / 1e9))
    print("\n=== Exercício 7 (256-QAM) ===")
    bits, per = 8, 10
    simb = f / per
    taxa = simb * bits
    print("{} bits/símbolo; 1 símbolo a cada {} períodos -> {:.2e} símbolos/s -> {:.2f} Gbit/s".format(
        bits, per, simb, taxa / 1e9))
    total = 480e6 * 8
    print("480 MB = {:.2e} bits -> {:.2f} s".format(total, total / taxa))


if __name__ == "__main__":
    ex1(); ex2(); ex3(); ex4(); ex5(); ex6_7()
    plt.show()
