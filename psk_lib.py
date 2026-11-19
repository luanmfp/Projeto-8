# Projeto 8 - Parte 2: biblioteca do chat com modulação digital BPSK (PSK de 2 fases)
#
#   bit 0 -> portadora com fase   0°   ->  A·sin(2π·fc·t)
#   bit 1 -> portadora com fase 180°   ->  A·sin(2π·fc·t + π)
#
# Quadro transmitido:  [preâmbulo 31 bits][comprimento 8 bits][mensagem UTF-8][CRC-8 8 bits]
import numpy as np
from suaBibSignal import signalMeu

FS = 44100        # taxa de amostragem (Hz)
FC = 4000         # portadora (Hz): deve ficar entre 2 kHz e 6 kHz
BPS = 100         # taxa de transmissão (bits/s)
AMP = 0.8         # amplitude da portadora
SILENCIO = 0.3    # s de silêncio antes e depois do quadro

sinal = signalMeu()


# ------------------------------------------------------------------ bits, preâmbulo e CRC
def sequencia_m31():
    """Sequência-m de 31 bits (LFSR x^5 + x^3 + 1). Autocorrelação muito boa:
    só "casa" consigo mesma quando perfeitamente alinhada -> serve para achar o início do quadro."""
    reg = [1, 0, 0, 1, 0]
    saida = []
    for _ in range(31):
        saida.append(reg[4])
        reg = [reg[4] ^ reg[2]] + reg[:4]
    return saida


PREAMBULO = sequencia_m31()


def bytes_para_bits(bs):
    return [(b >> i) & 1 for b in bs for i in range(7, -1, -1)]       # MSB primeiro


def bits_para_bytes(bits):
    return bytes(int("".join(str(b) for b in bits[i:i + 8]), 2) for i in range(0, len(bits) - 7, 8))


def crc8(dados):
    crc = 0
    for byte in dados:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def montar_quadro(texto):
    payload = texto.encode("utf-8")
    if len(payload) > 255:
        raise ValueError("Mensagem muito longa (máx. 255 bytes).")
    corpo = bytes([len(payload)]) + payload
    return PREAMBULO + bytes_para_bits(corpo + bytes([crc8(corpo)]))


# ------------------------------------------------------------------ modulador
def amostras_por_bit(fs=FS, bps=BPS):
    if fs % bps != 0:
        raise ValueError("fs/bps precisa ser inteiro (ex.: bps = 50, 100, 150, 300, 450, 900 para fs = 44100).")
    return fs // bps


def modular_bits(bits, fc=FC, bps=BPS, fs=FS, amp=AMP):
    n = amostras_por_bit(fs, bps)
    fase = np.repeat(np.array(bits) * np.pi, n)          # 0 rad para o bit 0, π rad para o bit 1
    t = np.arange(len(fase)) / fs
    return amp * np.sin(2 * np.pi * fc * t + fase)


def gerar_audio(texto, fc=FC, bps=BPS, fs=FS):
    bits = montar_quadro(texto)
    s = modular_bits(bits, fc, bps, fs)
    sil = np.zeros(int(SILENCIO * fs))
    return np.concatenate([sil, s, sil]), bits


# ------------------------------------------------------------------ demodulador
def decodificar(x, fc=FC, bps=BPS, fs=FS, qmin=0.5, mu=0.2):
    """Demodulação coerente de BPSK.
    1) y = x · e^{-j2π fc t} (traz a portadora para 0 Hz); z = soma de y em cada janela de 1 bit
       (filtro "casado": integra o bit inteiro). Para BPSK, z ≈ ±A·e^{jθ}.
    2) Correlaciona z com o preâmbulo conhecido: o máximo dá o INÍCIO do quadro e a fase θ da portadora.
    3) Para cada bit: v = z·e^{-jθ}; bit = 0 se Re(v) > 0, senão 1.
       Uma malha de Costas (ajuste de θ pelo erro de fase) acompanha pequenas diferenças de clock.
    Retorna um dicionário (ou None se não encontrou quadro)."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = amostras_por_bit(fs, bps)
    L = len(PREAMBULO)
    N = len(x)
    if N < (L + 24) * n:
        return None

    t = np.arange(N) / fs
    y = x * np.exp(-2j * np.pi * fc * t)
    c = np.concatenate(([0], np.cumsum(y)))
    z = c[n:] - c[:-n]                                    # z[i] = soma de y[i : i+n]

    p = 1 - 2 * np.array(PREAMBULO)                       # bit 0 -> +1, bit 1 -> -1
    M = len(z) - (L - 1) * n
    if M <= 0:
        return None
    acc = np.zeros(M, dtype=complex)
    den = np.zeros(M)
    for k in range(L):
        seg = z[k * n:k * n + M]
        acc += p[k] * seg
        den += np.abs(seg)
    n0 = int(np.argmax(np.abs(acc)))
    qualidade = float(np.abs(acc[n0]) / (den[n0] + 1e-12))   # ~1 = preâmbulo perfeito; ~0 = só ruído
    if qualidade < qmin:
        return None
    theta0 = float(np.angle(acc[n0]))

    # demodula os bits seguintes
    theta = theta0
    K = (len(z) - 1 - n0) // n + 1                        # nº de símbolos disponíveis a partir de n0
    bits, total = [], None
    for k in range(L, K):
        v = z[n0 + k * n] * np.exp(-1j * theta)
        bits.append(0 if v.real >= 0 else 1)
        theta += mu * np.arctan2(v.imag * np.sign(v.real), abs(v.real))   # malha de Costas
        if len(bits) == 8 and total is None:
            total = 8 + 8 * int(bits_para_bytes(bits)[0]) + 8
        if total is not None and len(bits) >= total:
            break

    info = dict(qualidade=qualidade, inicio_s=n0 / fs, theta0_graus=np.degrees(theta0),
                n_bits=len(bits), esperado=total, texto=None, crc_ok=False)
    if total is None or len(bits) < total:
        info["erro"] = "quadro incompleto"
        return info
    dados = bits_para_bytes(bits[:total])
    corpo, crc_rx = dados[:-1], dados[-1]
    info["crc_ok"] = crc8(corpo) == crc_rx
    info["comprimento"] = corpo[0]
    info["texto"] = corpo[1:].decode("utf-8", errors="replace")
    return info


# ------------------------------------------------------------------ análise espectral
def largura_banda(x, fs, fc, frac=0.99):
    """Menor banda SIMÉTRICA em torno de fc que contém 'frac' da potência do sinal (Hz)."""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    ordem = np.argsort(np.abs(f - fc))
    acum = np.cumsum(X[ordem]) / X.sum()
    d = np.abs(f[ordem][np.searchsorted(acum, frac)] - fc)
    return 2 * d


def plot_fft(ax, x, fs, fc, bps, titulo, span=None):
    xf, yf = sinal.calcFFT(x, fs)
    span = span or 10 * bps
    ax.plot(xf, yf)
    ax.set_xlim(fc - span, fc + span)
    ax.axvline(fc, color="k", linestyle="--", linewidth=0.8, label="portadora {} Hz".format(fc))
    ax.axvline(fc - bps, color="r", linestyle=":", label="fc ± R (1ºs nulos)")
    ax.axvline(fc + bps, color="r", linestyle=":")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
