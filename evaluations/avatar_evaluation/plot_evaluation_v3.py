#!/usr/bin/env python3
"""
audio_cue_check.py

Diagnostico rapido: existe cue visivel de mudanca emocional no eixo do audio?

Nao envolve video, nem py-feat, nem classificador de emocao. So prosodia.
A pergunta que este script responde e uma so: quando o audio muda de estado
emocional, algum descritor prosodico lento mostra um degrau?

Se a resposta for nao, nenhuma analise de correlacao face-audio baseada em
prosodia vai funcionar, e o eixo do audio precisa ser um modelo de emocao.

Descritores (todos agregados em janela deslizante centrada):
  1. F0 mediana        -> registro / pitch base
  2. F0 IQR            -> variabilidade de pitch (costuma discriminar melhor
                          que a mediana; raiva e alegria expandem o range)
  3. RMS p90           -> pico de energia, robusto a silencio
  4. Razao HF (>1 kHz) -> proxy de esforco vocal / tensao (arousal)

Uso:
  python audio_cue_check.py --audio fala.wav --out cue_check.png

  # com os limites de segmento do estimulo (recomendado):
  python audio_cue_check.py --audio fala.wav \
      --segments "0:4:neutro,4:11:feliz,11:16:neutro" \
      --out cue_check.png

  # varredura de tamanho de janela, para achar a escala temporal certa:
  python audio_cue_check.py --audio fala.wav --sweep --out cue_sweep.png
"""

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

ANALYSIS_SR = 16000     # reamostra tudo: 48 kHz nao acrescenta nada aqui e
                        # deixa o pyin lento sem necessidade
FRAME = 1024            # ~64 ms em 16 kHz
HOP = 256               # ~16 ms em 16 kHz
HF_CUTOFF = 1000.0      # Hz, fronteira para a razao de energia alta
SILENCE_REL = 0.10      # frames abaixo de 10% do p90 de RMS viram NaN

# Teto de F0. O default antigo (C7, 2093 Hz) e alto demais para fala e
# convida erro de oitava: o estimador prefere um pico espurio no dobro da
# frequencia. 400 Hz cobre voz feminina falada com folga.
F0_MIN_DEFAULT = 65.0
F0_MAX_DEFAULT = 400.0


# ---------------------------------------------------------------------------
# Extracao frame-level
# ---------------------------------------------------------------------------
def median_filter_nan(x, k=5):
    """Filtro de mediana que ignora NaN; suaviza saltos de oitava isolados."""
    n = len(x)
    out = np.full(n, np.nan)
    half = k // 2
    for i in range(n):
        w = x[max(0, i - half):min(n, i + half + 1)]
        w = w[~np.isnan(w)]
        if len(w):
            out[i] = np.median(w)
    return out



def frame_level_features(audio_path, f0_method="pyin",
                         fmin=F0_MIN_DEFAULT, fmax=F0_MAX_DEFAULT,
                         trim_start=5.0, trim_end=5.0):
    y_full, sr = librosa.load(audio_path, sr=ANALYSIS_SR, mono=True)
    dur_full = len(y_full) / sr

    # Recorta as bordas: onset de articulacao e cauda da ultima silaba geram
    # transientes que dominam a envelope e inflam o desvio dos segmentos
    # extremos. O eixo de tempo permanece no referencial do arquivo original
    # (offset = trim_start), para que segmentos e video continuem alinhados.
    i0 = int(round(trim_start * sr))
    i1 = len(y_full) - int(round(trim_end * sr))
    if i1 - i0 < sr:                        # sobrou menos de 1s
        raise SystemExit(
            f"\nERRO: apos recortar {trim_start:.2f}s do inicio e "
            f"{trim_end:.2f}s do fim, sobrou menos de 1s de audio "
            f"(duracao original {dur_full:.2f}s). Reduza --trim.\n")
    y = y_full[i0:i1]
    t_offset = i0 / sr

    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    t = librosa.times_like(rms, sr=sr, hop_length=HOP) + t_offset

    # mascara de silencio: usada para nao contaminar F0 e razao HF
    thr = SILENCE_REL * np.percentile(rms, 90)
    voiced_energy = rms > thr

    if f0_method == "pyin":
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax,
            sr=sr, frame_length=FRAME, hop_length=HOP,
        )
    else:
        # yin nao tem deteccao de sonoridade: devolve uma estimativa para todo
        # frame, inclusive fricativa, ruido e silencio residual. Sem gating,
        # a mediana e o IQR viram artefato. Aqui o gating e (a) energia,
        # (b) faixa plausivel, (c) filtro de mediana curto contra saltos de
        # oitava isolados. Ainda assim, prefira pyin para a figura final.
        f0 = librosa.yin(
            y, fmin=fmin, fmax=fmax,
            sr=sr, frame_length=FRAME, hop_length=HOP,
        )
        f0 = np.asarray(f0, dtype=float)
        f0[(f0 <= fmin * 1.02) | (f0 >= fmax * 0.98)] = np.nan
        f0 = median_filter_nan(f0, k=5)
        voiced_flag = ~np.isnan(f0)

    f0 = np.asarray(f0, dtype=float)
    f0[~voiced_flag] = np.nan
    f0[~voiced_energy[:len(f0)]] = np.nan

    # rejeicao final de outlier: fora de [p5/2, p95*2] da propria distribuicao
    valid = f0[~np.isnan(f0)]
    if len(valid) > 20:
        lo, hi = np.percentile(valid, 5) * 0.5, np.percentile(valid, 95) * 2.0
        f0[(f0 < lo) | (f0 > hi)] = np.nan

    # razao de energia acima de HF_CUTOFF
    S = np.abs(librosa.stft(y, n_fft=FRAME, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME)
    total = S.sum(axis=0)
    hi = S[freqs > HF_CUTOFF].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hf_ratio = np.where(total > 0, hi / total, np.nan)
    hf_ratio[~voiced_energy[:len(hf_ratio)]] = np.nan

    n = min(len(t), len(rms), len(f0), len(hf_ratio))
    return {
        "t": t[:n], "rms": rms[:n], "f0": f0[:n], "hf": hf_ratio[:n],
        "sr": sr,
        "dur": dur_full,                    # duracao do arquivo original
        "t0": t_offset,                     # inicio da regiao analisada
        "t1": t_offset + len(y) / sr,       # fim da regiao analisada
        "trim": (trim_start, trim_end),
    }


# ---------------------------------------------------------------------------
# Agregacao em janela deslizante centrada
# ---------------------------------------------------------------------------
def rolling(t, values, win, hop, fn):
    """Aplica fn a cada janela centrada de largura win, com passo hop."""
    centers = np.arange(t[0] + win / 2, t[-1] - win / 2 + 1e-9, hop)
    if len(centers) == 0:
        centers = np.array([(t[0] + t[-1]) / 2])
    out = np.full(len(centers), np.nan)
    half = win / 2
    for i, c in enumerate(centers):
        m = (t >= c - half) & (t <= c + half)
        chunk = values[m]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) >= 3:                 # evita estatistica de 1 amostra
            out[i] = fn(chunk)
    return centers, out


def envelopes(feat, win, hop):
    t = feat["t"]
    c, f0_med = rolling(t, feat["f0"], win, hop, np.median)
    _, f0_iqr = rolling(t, feat["f0"], win, hop,
                        lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    _, rms_p90 = rolling(t, feat["rms"], win, hop,
                         lambda x: np.percentile(x, 90))
    _, hf_med = rolling(t, feat["hf"], win, hop, np.median)
    return {
        "t": c,
        "F0 mediana (Hz)": f0_med,
        "F0 IQR (Hz)": f0_iqr,
        "RMS p90": rms_p90,
        "Razao HF (>1kHz)": hf_med,
    }


# ---------------------------------------------------------------------------
# Segmentos e tamanho de efeito
# ---------------------------------------------------------------------------
def parse_segments(spec):
    """'0:4:neutro,4:11:feliz' -> [(0.0, 4.0, 'neutro'), (4.0, 11.0, 'feliz')]"""
    if not spec:
        return []
    segs = []
    for part in spec.split(","):
        bits = part.split(":")
        if len(bits) != 3:
            raise ValueError(f"segmento mal formado: {part}")
        segs.append((float(bits[0]), float(bits[1]), bits[2]))
    return segs


def validate_segments(segs, dur):
    """Aborta se um segmento cai fora do audio, em vez de devolver d = nan."""
    bad = [s for s in segs if s[0] >= dur]
    if bad:
        names = ", ".join(f"{s[2]} [{s[0]:.1f}-{s[1]:.1f}s]" for s in bad)
        raise SystemExit(
            f"\nERRO: o audio tem {dur:.2f}s, mas estes segmentos comecam depois "
            f"do fim do arquivo:\n  {names}\n"
            f"Confira a duracao do wav (e do mp4 correspondente) antes de "
            f"reinterpretar qualquer coisa.\n")
    clipped = []
    for s0, s1, lab in segs:
        if s1 > dur:
            warnings.warn(f"segmento '{lab}' truncado de {s1:.1f}s para {dur:.2f}s")
            s1 = dur
        clipped.append((s0, s1, lab))
    return clipped


# ---------------------------------------------------------------------------
# Busca de fronteira sem rotulo
# ---------------------------------------------------------------------------
def scan_split(env, edge=1.5, step=0.25):
    """Varre cortes candidatos e reporta o que maximiza |d| entre as metades.

    Serve quando voce nao sabe onde esta a transicao. ATENCAO: o corte e
    escolhido olhando os dados, entao o |d| resultante e otimista por
    construcao. Use para localizar a fronteira e comparar com a verdadeira,
    nunca como numero reportavel.
    """
    keys = [k for k in env if k != "t"]
    t = env["t"]
    cuts = np.arange(t[0] + edge, t[-1] - edge, step)
    if len(cuts) < 2:
        print("\nAudio curto demais para varredura de fronteira.\n")
        return {}

    print("\nVarredura de fronteira (corte ajustado aos dados, d otimista)")
    print("-" * 62)
    best = {}
    for k in keys:
        ds = []
        for c in cuts:
            d = cohens_d(env[k][t < c], env[k][t >= c])
            ds.append(0.0 if np.isnan(d) else abs(d))
        ds = np.array(ds)
        i = int(np.argmax(ds))
        best[k] = (cuts[i], ds[i])
        print(f"  {k:<22} melhor corte = {cuts[i]:5.2f}s   |d| = {ds[i]:.2f}")
    print("-" * 62)
    cs = [v[0] for v in best.values()]
    print(f"  mediana dos cortes: {np.median(cs):.2f}s   "
          f"(dispersao: {np.min(cs):.2f} a {np.max(cs):.2f}s)")
    print("  Cortes concordantes entre descritores sugerem uma fronteira real.")
    print("  Cortes espalhados sugerem que nao ha transicao prosodica.\n")
    return best


def cohens_d(a, b):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    return (a.mean() - b.mean()) / sp


def report_effect_sizes(env, segs):
    """d entre cada par de segmentos, por descritor.

    Nota: as amostras da envelope sao fortemente autocorrelacionadas (janelas
    sobrepostas), entao d aqui e descritivo, para ordenar descritores por
    separabilidade. Nao vale como teste de hipotese.
    """
    if len(segs) < 2:
        return
    keys = [k for k in env if k != "t"]
    t = env["t"]
    print("\nTamanho de efeito (Cohen's d) entre segmentos")
    print("-" * 62)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            s1, s2 = segs[i], segs[j]
            m1 = (t >= s1[0]) & (t < s1[1])
            m2 = (t >= s2[0]) & (t < s2[1])
            print(f"\n  {s1[2]} [{s1[0]:.1f}-{s1[1]:.1f}s]  vs  "
                  f"{s2[2]} [{s2[0]:.1f}-{s2[1]:.1f}s]")
            for k in keys:
                d = cohens_d(env[k][m1], env[k][m2])
                mark = ""
                if not np.isnan(d):
                    a = abs(d)
                    mark = "  <-- forte" if a >= 0.8 else ("  <-- medio" if a >= 0.5 else "")
                print(f"      {k:<22} d = {d:+.2f}{mark}")
    print("\n" + "-" * 62)
    print("Referencia: |d| >= 0.8 grande, 0.5 medio, 0.2 pequeno.")
    print("Se nenhum descritor passa de ~0.5, prosodia bruta nao serve como")
    print("eixo do audio e o proximo passo e uma curva de arousal de modelo.\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def mark_trim(ax, feat):
    """Sombreia as bordas descartadas, para o recorte ficar explicito na figura."""
    t0, t1, dur = feat["t0"], feat["t1"], feat["dur"]
    if t0 > 0:
        ax.axvspan(0, t0, color="0.6", alpha=0.35, zorder=4, lw=0, hatch="//")
    if t1 < dur:
        ax.axvspan(t1, dur, color="0.6", alpha=0.35, zorder=4, lw=0, hatch="//")


def shade_segments(ax, segs):
    palette = {"neutro": "0.85", "neutral": "0.85"}
    cycle = ["#ffd9b3", "#b3d9ff", "#c6f0c6", "#f0c6f0"]
    ci = 0
    for s0, s1, lab in segs:
        key = lab.lower()
        if key in palette:
            color = palette[key]
        else:
            color = cycle[ci % len(cycle)]
            ci += 1
        ax.axvspan(s0, s1, color=color, alpha=0.45, zorder=0, lw=0)
        ax.axvline(s0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.axvline(s1, color="0.4", lw=0.8, ls="--", zorder=1)


def plot_envelopes(feat, env, segs, win, out_path):
    keys = [k for k in env if k != "t"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.1 * len(keys)),
                             sharex=True)
    raw_for = {"F0 mediana (Hz)": ("f0", "tab:red"),
               "RMS p90": ("rms", "tab:blue")}

    for ax, k in zip(axes, keys):
        if segs:
            shade_segments(ax, segs)
        if k in raw_for:                    # sinal bruto ao fundo, para contraste
            src, col = raw_for[k]
            ax.plot(feat["t"], feat[src], color=col, lw=0.6, alpha=0.25, zorder=2)
        ax.plot(env["t"], env[k], color="k", lw=2.0, zorder=3)
        mark_trim(ax, feat)
        ax.set_xlim(0, feat["dur"])
        ax.set_ylabel(k, fontsize=9)
        ax.grid(alpha=0.25)

    axes[0].set_title(f"Cue prosodico lento  -  janela = {win:.1f}s "
                      f"(preto = envelope, colorido claro = bruto)", pad=18)
    if segs:
        for s0, s1, lab in segs:
            axes[0].text((s0 + s1) / 2, 1.02, lab,
                         transform=axes[0].get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=9, weight="bold",
                         clip_on=False)
    axes[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura salva em: {out_path}")


def plot_sweep(feat, segs, wins, hop, out_path):
    """Mesmo descritor sob varias janelas: acha a escala temporal do cue."""
    keys = ["F0 mediana (Hz)", "F0 IQR (Hz)", "RMS p90", "Razao HF (>1kHz)"]
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.1 * len(keys)),
                             sharex=True)
    for w in wins:
        env = envelopes(feat, w, hop)
        for ax, k in zip(axes, keys):
            ax.plot(env["t"], env[k], lw=1.6, label=f"{w:.1f}s")
    for ax, k in zip(axes, keys):
        if segs:
            shade_segments(ax, segs)
        mark_trim(ax, feat)
        ax.set_xlim(0, feat["dur"])
        ax.set_ylabel(k, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(title="janela", fontsize=8, ncol=len(wins), loc="upper right")
    axes[0].set_title("Varredura de tamanho de janela")
    axes[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figura salva em: {out_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Checa se ha cue prosodico visivel de mudanca emocional")
    p.add_argument("--audio", required=True)
    p.add_argument("--win", type=float, default=1.5,
                   help="janela da envelope em segundos (default 1.5)")
    p.add_argument("--hop", type=float, default=0.1,
                   help="passo da envelope em segundos (default 0.1)")
    p.add_argument("--segments", default=None,
                   help="'inicio:fim:rotulo,...' em segundos")
    p.add_argument("--f0-method", choices=["pyin", "yin"], default="pyin",
                   help="pyin e o default: yin nao detecta sonoridade e "
                        "distorce mediana e IQR de F0")
    p.add_argument("--fmin", type=float, default=F0_MIN_DEFAULT)
    p.add_argument("--fmax", type=float, default=F0_MAX_DEFAULT,
                   help="teto de F0 (default 400 Hz; acima disso ha erro de oitava)")
    p.add_argument("--scan-split", action="store_true",
                   help="varre cortes candidatos quando a fronteira e desconhecida")
    p.add_argument("--trim", type=float, default=0.5,
                   help="segundos descartados de CADA borda do audio "
                        "(default 0.5; use 0 para desativar)")
    p.add_argument("--trim-start", type=float, default=None,
                   help="sobrescreve --trim so no inicio")
    p.add_argument("--trim-end", type=float, default=None,
                   help="sobrescreve --trim so no fim")
    p.add_argument("--sweep", action="store_true",
                   help="plota varias janelas sobrepostas em vez de uma")
    p.add_argument("--sweep-wins", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--out", default="audio_cue_check.png")
    args = p.parse_args()

    segs = parse_segments(args.segments)

    ts = args.trim if args.trim_start is None else args.trim_start
    te = args.trim if args.trim_end is None else args.trim_end

    print(f"Extraindo features frame-level ({args.f0_method})...")
    feat = frame_level_features(args.audio, args.f0_method,
                                fmin=args.fmin, fmax=args.fmax,
                                trim_start=ts, trim_end=te)
    print(f"  duracao do arquivo: {feat['dur']:.2f}s, "
          f"sr de analise: {feat['sr']} Hz")
    if ts or te:
        print(f"  recorte de borda: -{ts:.2f}s no inicio, -{te:.2f}s no fim  "
              f"-> regiao analisada {feat['t0']:.2f}s a {feat['t1']:.2f}s")
    print(f"  frames: {len(feat['t'])}")
    voiced = np.mean(~np.isnan(feat["f0"])) * 100
    f0v = feat["f0"][~np.isnan(feat["f0"])]
    if len(f0v):
        print(f"  frames com F0 valido: {voiced:.1f}%  |  "
              f"F0 mediana global: {np.median(f0v):.0f} Hz  "
              f"(p5 {np.percentile(f0v, 5):.0f}, p95 {np.percentile(f0v, 95):.0f})")
        if np.median(f0v) > 350:
            warnings.warn("Mediana global de F0 acima de 350 Hz: suspeite de "
                          "erro de oitava e reduza --fmax.")
    if voiced < 20:
        warnings.warn("Pouquissimo F0 valido; a envelope de pitch sera instavel.")

    segs = validate_segments(segs, feat["dur"])

    if args.sweep:
        plot_sweep(feat, segs, args.sweep_wins, args.hop, args.out)
        env = envelopes(feat, args.win, args.hop)
    else:
        env = envelopes(feat, args.win, args.hop)
        plot_envelopes(feat, env, segs, args.win, args.out)

    report_effect_sizes(env, segs)

    if args.scan_split or not segs:
        scan_split(env)


if __name__ == "__main__":
    main()


'''

python plot_evaluation_v3.py \
    --audio data/id02_p01_e02_f23.wav \
    --video-dynamic data/stage3_id02_p01_e01_f23_dynamic.mp4 \
    --video-constant data/stage3_id02_p01_e01_f23_constant.mp4 \
    --out figura_id02_p01_e02_f23.png


python plot_evaluation_v3.py --audio data/id02_p01_e02_f23.wav --scan-split --out cue_check_id02_p01_e02_f23.png --trim 1.0

'''