# Projeto 8 - Parte 2: lado RECEPTOR do chat (BPSK a 100 bit/s)
#   python chat_receptor.py                 -> escuta continuamente e imprime as mensagens
#   python chat_receptor.py --fc 3000 --bps 150 --plot     (parâmetros iguais aos do emissor!)
import argparse
import queue
import collections
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi
import psk_lib as psk


def mostrar(audio, a, n_msg):
    info = psk.decodificar(audio, a.fc, a.bps)
    if info is None:
        print("[sinal detectado, mas nenhum quadro válido foi encontrado]")
        return
    print("  início do quadro: {:.3f} s | qualidade do preâmbulo: {:.2f} | fase θ0: {:.1f}° | bits: {}/{}".format(
        info["inicio_s"], info["qualidade"], info["theta0_graus"], info["n_bits"], info["esperado"]))
    if info["texto"] is None:
        print("  ", info.get("erro"))
        return
    print(">>> {}{}".format(info["texto"], "" if info["crc_ok"] else "   [CRC FALHOU: mensagem pode ter erros]"))

    bw = psk.largura_banda(audio, psk.FS, a.fc, 0.90)
    print("  banda (90% da potência) em torno de fc: {:.0f} Hz  (teórico 2·R = {} Hz)".format(bw, 2 * a.bps))
    fig, ax = plt.subplots(figsize=(9, 4))
    psk.plot_fft(ax, audio, psk.FS, a.fc, a.bps, "FFT do sinal RECEBIDO")
    fig.tight_layout()
    fig.savefig("fft_recebido.png", dpi=120)
    if a.plot:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", type=float, default=psk.FC)
    ap.add_argument("--bps", type=int, default=psk.BPS)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    bloco = int(0.05 * psk.FS)                                   # blocos de 50 ms
    meia = max(300, 2 * a.bps)
    sos = butter(4, [a.fc - meia, a.fc + meia], btype="band", fs=psk.FS, output="sos")
    zi = sosfilt_zi(sos) * 0
    q = queue.Queue()

    def cb(indata, frames, tempo, status):
        q.put(indata[:, 0].copy())

    print("Receptor BPSK | portadora {} Hz | {} bit/s".format(a.fc, a.bps))
    print("Calibrando o ruído (1 s, fique em silêncio)...")
    pre = collections.deque(maxlen=6)                            # 0,3 s de "pré-rolagem"
    ruidos, gravando, buf, silencio, n_msg = [], False, [], 0, 0
    limiar = None
    with sd.InputStream(samplerate=psk.FS, channels=1, blocksize=bloco, callback=cb):
        while True:
            x = q.get()
            f, zi = sosfilt(sos, x, zi=zi)
            rms = float(np.sqrt(np.mean(f ** 2)))
            if limiar is None:
                ruidos.append(rms)
                pre.append(x)
                if len(ruidos) >= 20:
                    limiar = max(4 * np.median(ruidos), 1e-3)
                    print("Pronto. Limiar de detecção: {:.4f}. Aguardando mensagens (Ctrl+C para sair).".format(limiar))
                continue
            if not gravando:
                pre.append(x)
                if rms > limiar:
                    gravando, buf, silencio = True, list(pre), 0
                    print("\n[recebendo...]")
            else:
                buf.append(x)
                silencio = silencio + 1 if rms < limiar else 0
                if silencio >= 8 or len(buf) > 600:               # 0,4 s sem sinal (ou 30 s) -> fim
                    gravando = False
                    n_msg += 1
                    mostrar(np.concatenate(buf), a, n_msg)
                    pre.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEncerrado.")
