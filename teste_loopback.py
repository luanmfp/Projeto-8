# Projeto 8 - Parte 2: teste SEM placa de som (canal simulado) + experimentos de taxa e portadora.
# Gera os gráficos em saida_parte2/ e imprime a banda medida em cada caso.
import os
import numpy as np
import matplotlib.pyplot as plt
import psk_lib as psk

PASTA = "saida_parte2"
os.makedirs(PASTA, exist_ok=True)
rng = np.random.default_rng(1)


def canal(x, fs, atenuacao=0.3, ruido=0.02, atraso_s=0.7, ppm=0.0):
    """Simula o ar entre alto-falante e microfone: atraso, atenuação, ruído e diferença de clock."""
    ini = rng.normal(0, ruido, int(atraso_s * fs))
    meio = x * atenuacao + rng.normal(0, ruido, len(x))
    fim = rng.normal(0, ruido, int(0.5 * fs))
    y = np.concatenate([ini, meio, fim])
    if ppm:                                                     # clock do receptor diferente do emissor
        t_novo = np.arange(len(y)) * (1 + ppm * 1e-6)
        y = np.interp(t_novo, np.arange(len(y)), y)
    return y


def teste_canal():
    print("=== Teste de transmissão/recepção (canal simulado) ===")
    msgs = ["Oi!", "Olá, mundo! Chat PSK funcionando.", "Camada Física da Computação – Insper 2026 ção é ñ"]
    ok = tot = 0
    for ruido, ppm in [(0.005, 0), (0.05, 0), (0.05, 100), (0.15, 50), (0.3, 0)]:
        for m in msgs:
            audio, _ = psk.gerar_audio(m)
            rx = canal(audio, psk.FS, ruido=ruido, ppm=ppm)
            info = psk.decodificar(rx)
            acerto = info is not None and info["texto"] == m and info["crc_ok"]
            ok += acerto; tot += 1
            print("ruído σ={:<5} clock {:>3} ppm | {:42s} -> {}".format(
                ruido, ppm, repr(m)[:42], "OK" if acerto else "FALHOU ({})".format(None if info is None else info.get("texto"))))
    print("Acertos: {}/{}".format(ok, tot))


def figura_emitido_recebido():
    texto = "Olá, mundo! Chat PSK funcionando."
    audio, bits = psk.gerar_audio(texto)
    rx = canal(audio, psk.FS, ruido=0.03, ppm=50)
    info = psk.decodificar(rx)
    print("\nDecodificado:", info)
    emit = audio[int(psk.SILENCIO * psk.FS):-int(psk.SILENCIO * psk.FS)]

    fig, axs = plt.subplots(3, 1, figsize=(9, 10))
    psk.plot_fft(axs[0], emit, psk.FS, psk.FC, psk.BPS, "FFT do sinal EMITIDO (BPSK 100 bit/s, fc = 4 kHz)")
    psk.plot_fft(axs[1], rx, psk.FS, psk.FC, psk.BPS, "FFT do sinal RECEBIDO (com ruído e atenuação)")
    n = psk.amostras_por_bit()
    L = len(psk.PREAMBULO)
    k = next(i for i in range(L, len(bits) - 1) if bits[i] != bits[i + 1])   # 1ª troca de bit nos dados
    centro = (k + 1) * n                                                   # instante da troca (em emit)
    ini, fim = centro - int(0.0025 * psk.FS), centro + int(0.0025 * psk.FS)
    t = np.arange(ini, fim) / psk.FS * 1000
    axs[2].plot(t, emit[ini:fim])
    axs[2].axvline(centro / psk.FS * 1000, color="r", linestyle=":", label="troca de bit {} -> {}".format(bits[k], bits[k + 1]))
    axs[2].legend(loc="upper right")
    axs[2].set_title("Sinal emitido no tempo: a fase inverte (180°) quando o bit muda")
    axs[2].set_xlabel("Tempo (ms)"); axs[2].set_ylabel("Amplitude"); axs[2].grid(True, alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(PASTA, "psk_emitido_recebido.png"), dpi=120)
    print("Banda (90% da potência) emitido: {:.0f} Hz | 99%: {:.0f} Hz | teórico 2·R = {} Hz".format(
        psk.largura_banda(emit, psk.FS, psk.FC, 0.90), psk.largura_banda(emit, psk.FS, psk.FC, 0.99), 2 * psk.BPS))


def experimento_taxas():
    print("\n=== Variando a taxa de transmissão (fc = 4000 Hz) ===")
    print("  bit/s | banda 90% (Hz) | banda 99% (Hz) | teórico 2·R")
    taxas = [50, 100, 150, 300]
    fig, axs = plt.subplots(len(taxas), 1, figsize=(9, 10), sharex=True)
    for ax, bps in zip(axs, taxas):
        bits = list(rng.integers(0, 2, 3000 * bps // 50 // 3))
        s = psk.modular_bits(bits, 4000, bps)
        print("  {:5d} | {:14.0f} | {:14.0f} | {:6d}".format(
            bps, psk.largura_banda(s, psk.FS, 4000, 0.90), psk.largura_banda(s, psk.FS, 4000, 0.99), 2 * bps))
        psk.plot_fft(ax, s, psk.FS, 4000, bps, "{} bit/s".format(bps), span=800)
    fig.tight_layout(); fig.savefig(os.path.join(PASTA, "psk_taxas.png"), dpi=120)


def experimento_portadoras():
    print("\n=== Variando a portadora (100 bit/s) ===")
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    bits = list(rng.integers(0, 2, 1500))
    for ax, fc in zip(axs, [2000, 4000, 6000]):
        s = psk.modular_bits(bits, fc, 100)
        xf, yf = psk.sinal.calcFFT(s, psk.FS)
        ax.plot(xf, yf); ax.set_xlim(0, 8000); ax.axvline(fc, color="k", linestyle="--", linewidth=0.8)
        ax.set_title("fc = {} Hz".format(fc)); ax.grid(True, alpha=0.4); ax.set_ylabel("Amplitude")
        pico = xf[np.argmax(yf)]
        print("  fc = {} Hz -> pico em {:.0f} Hz, banda 90% = {:.0f} Hz".format(
            fc, pico, psk.largura_banda(s, psk.FS, fc, 0.90)))
    axs[-1].set_xlabel("Frequência (Hz)")
    fig.tight_layout(); fig.savefig(os.path.join(PASTA, "psk_portadoras.png"), dpi=120)


if __name__ == "__main__":
    teste_canal()
    figura_emitido_recebido()
    experimento_taxas()
    experimento_portadoras()
