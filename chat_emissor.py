# Projeto 8 - Parte 2: lado EMISSOR do chat (BPSK a 100 bit/s)
#   python chat_emissor.py                 -> fc = 4000 Hz, 100 bit/s
#   python chat_emissor.py --fc 3000 --bps 150 --plot
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import psk_lib as psk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", type=float, default=psk.FC, help="portadora em Hz (2000 a 6000)")
    ap.add_argument("--bps", type=int, default=psk.BPS, help="taxa em bit/s (divisor de 44100: 50, 100, 150, 300...)")
    ap.add_argument("--plot", action="store_true", help="abre o gráfico da FFT a cada mensagem")
    a = ap.parse_args()
    if not 2000 <= a.fc <= 6000:
        print("Aviso: o enunciado pede portadora entre 2 kHz e 6 kHz.")

    print("Emissor BPSK | portadora {} Hz | {} bit/s | {} amostras/bit".format(
        a.fc, a.bps, psk.amostras_por_bit(psk.FS, a.bps)))
    while True:
        texto = input("\nDigite a mensagem (ou 'sair'): ")
        if texto.strip().lower() == "sair":
            break
        if not texto:
            continue
        audio, bits = psk.gerar_audio(texto, a.fc, a.bps)
        dur = len(bits) / a.bps
        print("Bits do quadro: {} (preâmbulo {} + dados {}) -> {:.2f} s de áudio".format(
            len(bits), len(psk.PREAMBULO), len(bits) - len(psk.PREAMBULO), dur))
        print("Primeiros bits dos dados:", bits[len(psk.PREAMBULO):len(psk.PREAMBULO) + 16])
        print("Banda necessária ≈ 2·R = {} Hz  ({:.0f} Hz a {:.0f} Hz)".format(2 * a.bps, a.fc - a.bps, a.fc + a.bps))
        print("Transmitindo...")
        sd.play(audio, psk.FS)

        fig, ax = plt.subplots(figsize=(9, 4))
        sinal_modulado = audio[int(psk.SILENCIO * psk.FS):-int(psk.SILENCIO * psk.FS)]
        psk.plot_fft(ax, sinal_modulado, psk.FS, a.fc, a.bps, "FFT do sinal EMITIDO")
        fig.tight_layout()
        fig.savefig("fft_emitido.png", dpi=120)
        sd.wait()
        print("Enviado. (gráfico salvo em fft_emitido.png)")
        if a.plot:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
