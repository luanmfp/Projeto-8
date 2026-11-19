# Projeto 8 - Parte 1: 3 estações de rádio AM (modulação, soma no "meio físico",
# filtragem passa-faixa no receptor e demodulação síncrona).
#
#   python projeto8_parte1.py              -> usa audio1.wav, audio2.wav, audio3.wav (se faltar, gera áudios demo)
#   python projeto8_parte1.py --sem-audio  -> não toca os áudios (só gráficos e arquivos)
#
# Dependências: numpy, scipy, matplotlib, sounddevice (e suaBibSignal.py)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly
from suaBibSignal import signalMeu

# ------------------------------------------------------------------ parâmetros
FAIXAS = [(9000, 12000), (12000, 15000), (15000, 18000)]   # Hz (faixas 1, 2 e 3)
PORTADORAS = [(f1 + f2) / 2 for f1, f2 in FAIXAS]            # centro de cada faixa: 10,5 / 13,5 / 16,5 kHz
FM_MAX = 1400            # Hz: maior frequência do áudio. A banda de AM é [fc-fm, fc+fm] = 2*fm = 2,8 kHz
                         # (< 3 kHz da faixa, sobrando 100 Hz de guarda em cada lado)
ARQUIVOS = ["audio1.wav", "audio2.wav", "audio3.wav"]
PASTA = "saida_parte1"
sinal = signalMeu()


# ------------------------------------------------------------------ áudio
def carregar_audio(caminho):
    fs, x = wavfile.read(caminho)
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(float) - 128) / 128.0
    else:
        x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)           # estéreo -> mono
    return fs, x


def gerar_audios_demo(fs=44100, dur=4):
    """Áudios sintéticos (usados só se você ainda não gravou os seus)."""
    t = np.arange(int(fs * dur)) / fs
    a1 = np.sin(2 * np.pi * (300 + 100 * np.sin(2 * np.pi * 0.7 * t)) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    a2 = np.sin(2 * np.pi * 600 * t) + 0.5 * np.sin(2 * np.pi * 900 * t) * np.sign(np.sin(2 * np.pi * 1.5 * t))
    a3 = np.sin(2 * np.pi * (500 + 700 * t / dur) * t) + 0.4 * np.sin(2 * np.pi * 1100 * t)
    for nome, a in zip(ARQUIVOS, (a1, a2, a3)):
        wavfile.write(nome, fs, (a / np.max(np.abs(a)) * 0.8 * 32767).astype(np.int16))
    print("Áudios de demonstração gerados:", ARQUIVOS)


def carregar_tres():
    if not all(os.path.exists(a) for a in ARQUIVOS):
        print("Arquivos de áudio não encontrados.")
        gerar_audios_demo()
    fs, xs = None, []
    for a in ARQUIVOS:
        f, x = carregar_audio(a)
        if fs is None:
            fs = f
        elif f != fs:                                    # iguala as taxas de amostragem
            x = resample_poly(x, fs, f)
        xs.append(x)
    n = min(len(x) for x in xs)                          # mesmo número de pontos nos 3 sinais
    return fs, [x[:n] for x in xs]


def tocar(x, fs, rotulo, ativo=True):
    x = np.asarray(x)
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.9
    if not ativo:
        return
    try:
        import sounddevice as sd
        print("Tocando:", rotulo)
        sd.play(x, fs)
        sd.wait()
    except Exception as e:
        print("(sem áudio: {})".format(e))


# ------------------------------------------------------------------ processamento
def passa_baixa(x, fc, fs, ordem=8):
    sos = butter(ordem, fc, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)           # fase zero: não deforma o sinal


def passa_faixa(x, f1, f2, fs, ordem=8):
    sos = butter(ordem, [f1, f2], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def preparar_mensagem(x, fs, fm=FM_MAX):
    """Trata o áudio ANTES de modular: tira o nível DC e limita a banda a fm,
    para que a banda modulada [fc-fm, fc+fm] caiba na faixa da estação."""
    x = x - np.mean(x)
    x = passa_baixa(x, fm, fs)
    return x / np.max(np.abs(x))


def portadora(fc, n, fs):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * fc * t)


def modular(m, fc, fs):
    """AM por multiplicação: S(t) = m(t) * C(t)."""
    return m * portadora(fc, len(m), fs)


def demodular(s, fc, fs, fm=FM_MAX):
    """Demodulação síncrona: multiplica pela portadora e filtra passa-baixa.
    m*sin² = m/2 - (m/2)cos(4π fc t) -> o fator 2 recompõe a amplitude."""
    return passa_baixa(2 * s * portadora(fc, len(s), fs), fm, fs)


def fracao_fora(s, fs, f1, f2):
    """Fração da energia espectral fora de [f1, f2]."""
    xf, yf = sinal.calcFFT(s, fs)
    p = yf ** 2
    dentro = (xf >= f1) & (xf <= f2)
    return 1 - p[dentro].sum() / p.sum()


def correlacao(a, b):
    return np.corrcoef(a, b)[0, 1]


# ------------------------------------------------------------------ gráficos
def plot_fft(ax, x, fs, titulo, faixa=None, fmax=None):
    xf, yf = sinal.calcFFT(x, fs)
    ax.plot(xf / 1000, yf)
    ax.set_xlim(0, (fmax or fs / 2) / 1000)
    ax.set_xlabel("Frequência (kHz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.4)
    if faixa:
        ax.axvspan(faixa[0] / 1000, faixa[1] / 1000, color="orange", alpha=0.2, label="faixa permitida")
        ax.legend(loc="upper left")


def fig_tres(sinais, fs, titulo, nome_png, fmax=22050):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, (ax, s) in enumerate(zip(axs, sinais)):
        plot_fft(ax, s, fs, "Estação {} - faixa {}–{} kHz".format(i + 1, FAIXAS[i][0] // 1000, FAIXAS[i][1] // 1000),
                 FAIXAS[i], fmax)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, nome_png), dpi=120)
    return fig


# ------------------------------------------------------------------ programa
def main():
    tocar_audio = "--sem-audio" not in sys.argv
    os.makedirs(PASTA, exist_ok=True)

    fs, originais = carregar_tres()
    n = len(originais[0])
    print("fs = {} Hz, {} amostras ({:.1f} s) por áudio".format(fs, n, n / fs))
    print("Portadoras:", PORTADORAS, "Hz | banda de cada estação: 2*{} = {} Hz".format(FM_MAX, 2 * FM_MAX))

    # ---------------- 1) áudios originais
    for i, x in enumerate(originais):
        tocar(x, fs, "áudio original {}".format(i + 1), tocar_audio)

    # ---------------- 2) EMISSOR: trata, modula e verifica as bandas
    mensagens = [preparar_mensagem(x, fs) for x in originais]
    modulados = [modular(m, fc, fs) for m, fc in zip(mensagens, PORTADORAS)]
    print("\nEnergia fora da faixa de cada sinal modulado:")
    for i, (s, (f1, f2)) in enumerate(zip(modulados, FAIXAS)):
        print("  estação {}: {:.4%} fora de {}–{} Hz".format(i + 1, fracao_fora(s, fs, f1, f2), f1, f2))

    # ---------------- 3) MEIO FÍSICO: soma das 3 estações
    meio = modulados[0] + modulados[1] + modulados[2]

    # ---------------- 4) RECEPTOR: passa-faixa (sintoniza) + demodulação
    extraidos, demodulados = [], []
    for (f1, f2), fc in zip(FAIXAS, PORTADORAS):
        e = passa_faixa(meio, f1 + 50, f2 - 50, fs)      # "sintonizar" a estação
        extraidos.append(e)
        demodulados.append(demodular(e, fc, fs))

    print("\nQualidade da recuperação (correlação com a mensagem tratada de cada estação):")
    print("           msg1    msg2    msg3")
    for i, d in enumerate(demodulados):
        print("  demod {}: ".format(i + 1) + "  ".join("{:6.3f}".format(correlacao(d, m)) for m in mensagens))
    print("(diagonal ≈ 1 e fora da diagonal ≈ 0 => estações separadas sem interferência)")

    # ---------------- 5) gráficos pedidos
    fig_tres(modulados, fs, "FFT dos 3 sinais modulados (emissor)", "1_fft_modulados.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_fft(ax, meio, fs, "FFT do sinal resultante (soma das 3 estações no meio físico)")
    for f1, f2 in FAIXAS:
        ax.axvline(f1 / 1000, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(18, color="gray", linestyle=":", linewidth=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(PASTA, "2_fft_soma.png"), dpi=120)

    fig_tres(extraidos, fs, "FFT dos 3 sinais extraídos do sinal resultante (receptor, após passa-faixa)",
             "3_fft_extraidos.png")

    fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for i in range(3):
        plot_fft(axs[i, 0], originais[i], fs, "Áudio original {}".format(i + 1), fmax=4000)
        plot_fft(axs[i, 1], demodulados[i], fs, "Áudio demodulado {}".format(i + 1), fmax=4000)
    fig.tight_layout(); fig.savefig(os.path.join(PASTA, "4_fft_original_vs_demodulado.png"), dpi=120)

    # ---------------- 6) salva e toca os demodulados
    for i, d in enumerate(demodulados):
        wavfile.write(os.path.join(PASTA, "audio{}_demodulado.wav".format(i + 1)), fs,
                      (d / np.max(np.abs(d)) * 0.9 * 32767).astype(np.int16))
    print("\nGráficos e áudios demodulados salvos em '{}/'".format(PASTA))

    if tocar_audio:
        plt.show(block=False)
        plt.pause(0.5)
        for i, d in enumerate(demodulados):
            tocar(d, fs, "áudio DEMODULADO {}".format(i + 1), True)
    plt.show()


if __name__ == "__main__":
    main()
