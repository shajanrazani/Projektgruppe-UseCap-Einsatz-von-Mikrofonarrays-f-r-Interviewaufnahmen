import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, get_window, stft, istft

# =========================
# TUNING-PARAMETER
# =========================
FRAME_MS     = 40.0    # Frame-Länge für das Gating (stabiler als 20-30 ms)
HOP_MS       = 10.0    # Schrittweite
MIN_DBFS     = -48.0   # VAD: unter diesem Pegel -> Stille
MIN_SNR_DB   = 3.0     # Mindest-SNR (pro Frame/Kanal), sonst Stille/kein Mix
SNR_WITHIN_DB= 2.0     # Zweiter Kanal wird zugelassen, wenn SNR innerhalb dieser dB von Top1 liegt
RELEASE_FR   = 4       # Hysterese: Haltezeit in Frames (stabilisiert Umschalten)
HPF_HZ       = 80.0    # Hochpass gegen Dumpfheit
TARGET_PEAK  = 0.98    # Peak-Normalisierung
# Wiener-Postfilter (sehr mild)
WF_NPERSEG   = 640
WF_NOVERLAP  = 480
WF_FLOOR_DB  = -12.0   # minimaler Gain (dB)

# =========================
# Hilfsfunktionen
# =========================
def ensure_mono(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)

def butter_highpass(x, sr, fc=80.0, order=2):
    b, a = butter(order, fc / (0.5 * sr), btype='highpass')
    return lfilter(b, a, x)

def peak_normalize(x, target_peak=0.98):
    peak = np.max(np.abs(x)) + 1e-12
    if peak > target_peak:
        x = x * (target_peak / peak)
    return x

def frame_signal(x, frame_len, hop):
    T = len(x)
    if T < frame_len:
        x = np.pad(x, (0, frame_len - T))
        T = len(x)
    n = 1 + (T - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + np.arange(n)[:, None] * hop
    return x[idx], n * hop + frame_len - hop

def ola(recon_frames, win, hop, out_len):
    frame_len = recon_frames.shape[1]
    y = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    for i, fr in enumerate(recon_frames):
        t0 = i * hop
        t1 = t0 + frame_len
        y[t0:t1] += fr * win
        wsum[t0:t1] += win
    y /= np.maximum(wsum, 1e-12)
    return y

def rms(x, axis=-1, eps=1e-12):
    return np.sqrt(np.mean(np.square(x), axis=axis) + eps)

def dbfs(x, eps=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))

# =========================
# Rausch-Schätzer (pro Kanal)
# =========================
def estimate_noise_rms_per_channel(X, sr, frame_ms=40.0, hop_ms=10.0, perc=0.2):
    """
    Schätzt pro Kanal eine Rauschleistung (RMS) aus dem unteren Pegel-Quantil.
    X: [C,T]
    """
    C, T = X.shape
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    noise_rms = np.zeros(C, dtype=np.float32)

    for c in range(C):
        frames, _ = frame_signal(X[c], frame_len, hop)  # [N, L]
        fr_rms = rms(frames, axis=1)                   # [N]
        k = max(1, int(len(fr_rms) * perc))
        # Mittel der k leisesten Frames
        idx = np.argpartition(fr_rms, k)[:k]
        noise_rms[c] = np.mean(fr_rms[idx])
    return noise_rms  # [C]

# =========================
# Kern: SNR-basiertes Gating (+ optionaler 2. Kanal)
# =========================
def snr_gating_mixer(X, sr,
                     frame_ms=40.0, hop_ms=10.0,
                     min_dbfs=-48.0, min_snr_db=3.0, snr_within_db=2.0,
                     release_frames=4):
    """
    X: [C,T] (float32, normalisiert)
    Rückgabe: y (float32)
    """
    C, T = X.shape
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    win = get_window("hann", frame_len, fftbins=False).astype(np.float32)

    # Framing pro Kanal
    frames = []
    recon_len = None
    for c in range(C):
        fr, out_len = frame_signal(X[c], frame_len, hop)
        frames.append(fr)  # [N, L]
        recon_len = out_len
    N = frames[0].shape[0]

    # Pegel (dBFS) zur VAD
    db = np.zeros((C, N), dtype=np.float32)
    for c in range(C):
        db[c] = dbfs(rms(frames[c], axis=1))

    # Rauschschätzung (RMS) pro Kanal -> SNR pro Frame/Kanal
    noise_r = estimate_noise_rms_per_channel(X, sr, frame_ms, hop_ms, perc=0.2) + 1e-12
    # SNR in dB (aus RMS^2), limitiert nach unten
    snr_db = np.zeros((C, N), dtype=np.float32)
    for c in range(C):
        num = np.maximum(np.square(rms(frames[c], axis=1)) - noise_r[c]**2, 1e-12)
        den = noise_r[c]**2 + 1e-12
        snr_db[c] = 10.0 * np.log10(num / den)

    out_frames = np.zeros((N, frame_len), dtype=np.float32)

    # Hysterese-Zustand
    last_choice = (-1, -1)  # (top1, top2_or_-1)
    hold = 0

    for i in range(N):
        # Kandidaten nach SNR
        vals = snr_db[:, i]
        # minimale VAD-Pegelbedingung pro Kanal prüfen
        vad_ok = db[:, i] >= min_dbfs
        vals_eff = np.where(vad_ok, vals, -1e9)

        top1 = int(np.argmax(vals_eff))
        top1_snr = float(vals_eff[top1])

        if (not vad_ok[top1]) or (top1_snr < min_snr_db):
            # Stille
            last_choice = (-1, -1)
            hold = 0
            continue

        tmp = vals_eff.copy()
        tmp[top1] = -1e9
        top2 = int(np.argmax(tmp))
        top2_snr = float(tmp[top2])
        allow_two = (top2_snr >= (top1_snr - snr_within_db)) and vad_ok[top2] and (top2_snr >= min_snr_db)

        # Hysterese
        proposed = (top1, top2 if allow_two else -1)
        if proposed != last_choice and hold > 0:
            top1, top2 = last_choice[0], last_choice[1]
            allow_two = (top2 >= 0)
            hold -= 1
        else:
            last_choice = proposed
            hold = release_frames

        f1 = frames[top1][i]
        if allow_two:
            f2 = frames[top2][i]
            out_frames[i] = (f1 + f2) * (1.0 / np.sqrt(2.0))
        else:
            out_frames[i] = f1

    # Overlap-Add
    y = ola(out_frames, win=win, hop=hop, out_len=recon_len).astype(np.float32)
    return y

# =========================
# Leichter STFT-Wiener-Postfilter
# =========================
def mild_wiener_postfilter(y, sr, nperseg=512, noverlap=256, floor_db=-12.0):
    f, t, Y = stft(y, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, boundary=None)
    # Noise-Power aus leisesten 20% Frames des Betrags
    mag = np.abs(Y)
    frame_energy = np.mean(mag, axis=0)
    k = max(1, int(0.2 * len(frame_energy)))
    noise_idx = np.argpartition(frame_energy, k)[:k]
    Npsd = np.mean(np.abs(Y[:, noise_idx])**2, axis=1, keepdims=True) + 1e-12  # [F,1]

    # Signal-Power Schätzung
    Spsd = np.abs(Y)**2
    snr = Spsd / (Npsd + 1e-12)
    gain = snr / (snr + 1.0)
    gain = np.sqrt(np.clip(gain, 10.0**(floor_db/20.0), 1.0))

    Yf = Y * gain
    _, y_hat = istft(Yf, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=None)
    # Länge an Input angleichen
    if len(y_hat) < len(y):
        y_hat = np.pad(y_hat, (0, len(y)-len(y_hat)))
    elif len(y_hat) > len(y):
        y_hat = y_hat[:len(y)]
    return y_hat.astype(np.float32)

# =========================
# Main
# =========================
def main():
    in_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = in_dir

    # 4 Mics einsammeln (meeting_ch1..ch4), processed/ch5 ignorieren
    mic_wavs = sorted(
        w for w in glob.glob(os.path.join(in_dir, "BA_ch*.wav"))
        if ("ch0_processed" not in os.path.basename(w))
        and ("_ch5" not in os.path.basename(w))
    )
    if len(mic_wavs) < 2:
        raise RuntimeError(f"Zu wenige Mikrofondateien gefunden: {mic_wavs}")
    if len(mic_wavs) > 4:
        mic_wavs = mic_wavs[:4]

    print("Verwende Mikrofone (SNR-Gating):", mic_wavs)

    # Laden
    xs, srs = [], []
    for p in mic_wavs:
        x, sr = sf.read(p, always_2d=False)
        xs.append(ensure_mono(x))
        srs.append(sr)
    if len(set(srs)) != 1:
        raise ValueError(f"Samplerates uneinheitlich: {srs}")
    sr = srs[0]

    # Gleiche Länge & per-Kanal-Peak-Norm
    minlen = min(len(x) for x in xs)
    X = np.stack([x[:minlen] for x in xs], axis=0)  # [C,T]
    peak = np.max(np.abs(X), axis=1, keepdims=True) + 1e-12
    X = X / peak

    # Kern: SNR-basiertes Gating (mit optionaler 2. Stimme)
    y = snr_gating_mixer(
        X, sr,
        frame_ms=FRAME_MS, hop_ms=HOP_MS,
        min_dbfs=MIN_DBFS, min_snr_db=MIN_SNR_DB,
        snr_within_db=SNR_WITHIN_DB, release_frames=RELEASE_FR
    )

    # Entdumpfen + milder Wiener-Postfilter + Peak-Normalisierung
    y = butter_highpass(y, sr, fc=HPF_HZ, order=2)
    y = mild_wiener_postfilter(y, sr, nperseg=WF_NPERSEG, noverlap=WF_NOVERLAP, floor_db=WF_FLOOR_DB)
    y = peak_normalize(y, target_peak=TARGET_PEAK)

    out_path = os.path.join(out_dir, "outoput.wav")
    sf.write(out_path, y, sr)
    print("Fertig ->", out_path)

if __name__ == "__main__":
    main()
