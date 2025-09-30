import os
import glob
import time
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, get_window, stft, istft

# ============================================================
# KONFIG: Ordnerpfad & Betriebsarten
# ============================================================
IN_DIR = r""
GLOB_PATTERN = "uni_ch*_mic*.wav"  # erwartet ch1..ch4 ==> muss an eigenes Programm angepasst werden!

#Müssen aktiviert oder deaktiviert werden:
# ASR-Frontend (empfohlen für WER): minimal verzerrend
ASR_MODE = False            # True = kein Wiener-Postfilter, nur 1 Kanal (Top-SNR)
USE_DOA_DS = False         # Optional DoA+Delay-and-Sum (langsamer). Für WER erst nach ASR_MODE-Baseline testen!
MAX_SECONDS = None         # z.B. 60 für Tests, sonst None

# ============================================================
# TUNING-PARAMETER (aus deinem Code)
# ============================================================
FRAME_MS      = 40.0
HOP_MS        = 10.0
MIN_DBFS      = -48.0
MIN_SNR_DB    = 3.0
SNR_WITHIN_DB = 2.0        # wird ignoriert, wenn ASR_MODE=True (Single-Channel)
RELEASE_FR    = 4
HPF_HZ        = 80.0
TARGET_PEAK   = 0.98

# Wiener-Postfilter (nur wenn ASR_MODE=False)
WF_NPERSEG = 640
WF_NOVERLAP = 480
WF_FLOOR_DB = -12.0

# ============================================================
# DoA/Beamforming-Parameter (nur relevant, wenn USE_DOA_DS=True)
# ============================================================
SPEED_OF_SOUND = 343.0  # m/s
# Mikrofon-Positionen (Meter) in DERSELBEN REIHENFOLGE wie die Dateien!
MIC_POS = np.array([
    [-0.045, 0.0],  # Mic1
    [-0.015, 0.0],  # Mic2
    [ 0.015, 0.0],  # Mic3
    [ 0.045, 0.0],  # Mic4
], dtype=np.float32)

DOA_AZIMUTH_GRID_DEG = np.linspace(-90, 90, 181)  # ULA-Beispiel
DOA_MIN_CONF   = 0.20           # konservativer, um Fehlschätzungen zu vermeiden
DOA_ALPHA_SMOOTH = 0.90         # träge, stabil

# ============================================================
# Logging-Helper
# ============================================================
_t0_global = time.perf_counter()
def log(msg: str):
    now = time.perf_counter() - _t0_global
    print(f"[{now:7.2f}s] {msg}")

def seconds_to_str(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m {s:.1f}s"

# ============================================================
# Hilfsfunktionen
# ============================================================
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

# ============================================================
# Rausch-Schätzer (pro Kanal)
# ============================================================
def estimate_noise_rms_per_channel(X, sr, frame_ms=40.0, hop_ms=10.0, perc=0.2):
    C, T = X.shape
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    noise_rms = np.zeros(C, dtype=np.float32)
    for c in range(C):
        frames, _ = frame_signal(X[c], frame_len, hop)
        fr_rms = rms(frames, axis=1)
        k = max(1, int(len(fr_rms) * perc))
        idx = np.argpartition(fr_rms, k)[:k]
        noise_rms[c] = np.mean(fr_rms[idx])
    return noise_rms

# ============================================================
# Kern: SNR-basiertes Gating (mit Logs) — ASR-freundlich einstellbar
# ============================================================
def snr_gating_mixer(X, sr,
                     frame_ms=40.0, hop_ms=10.0,
                     min_dbfs=-48.0, min_snr_db=3.0, snr_within_db=2.0,
                     release_frames=4,
                     allow_two_channels=True,
                     stats_dict=None,
                     progress_callback=None):
    C, T = X.shape
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    win = get_window("hann", frame_len, fftbins=False).astype(np.float32)

    frames = []
    recon_len = None
    for c in range(C):
        fr, out_len = frame_signal(X[c], frame_len, hop)
        frames.append(fr)
        recon_len = out_len
    N = frames[0].shape[0]

    db = np.zeros((C, N), dtype=np.float32)
    for c in range(C):
        db[c] = dbfs(rms(frames[c], axis=1))

    noise_r = estimate_noise_rms_per_channel(X, sr, frame_ms, hop_ms, perc=0.2) + 1e-12
    snr_db = np.zeros((C, N), dtype=np.float32)
    for c in range(C):
        num = np.maximum(np.square(rms(frames[c], axis=1)) - noise_r[c]**2, 1e-12)
        den = noise_r[c]**2 + 1e-12
        snr_db[c] = 10.0 * np.log10(num / den)

    out_frames = np.zeros((N, frame_len), dtype=np.float32)
    last_choice = (-1, -1)
    hold = 0

    # Stats
    voiced = 0
    twoch = 0
    snr_top1_acc = 0.0

    for i in range(N):
        vals = snr_db[:, i]
        vad_ok = db[:, i] >= min_dbfs
        vals_eff = np.where(vad_ok, vals, -1e9)

        top1 = int(np.argmax(vals_eff))
        top1_snr = float(vals_eff[top1])

        if (not vad_ok[top1]) or (top1_snr < min_snr_db):
            last_choice = (-1, -1)
            hold = 0
            continue

        voiced += 1
        snr_top1_acc += top1_snr

        tmp = vals_eff.copy()
        tmp[top1] = -1e9
        top2 = int(np.argmax(tmp))
        top2_snr = float(tmp[top2])
        allow_two = (allow_two_channels and
                     (top2_snr >= (top1_snr - snr_within_db)) and
                     vad_ok[top2] and (top2_snr >= min_snr_db))

        if allow_two:
            twoch += 1

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

        if (progress_callback is not None) and (i % 500 == 0):
            progress_callback(i, N)

    y = ola(out_frames, win=win, hop=hop, out_len=recon_len).astype(np.float32)

    if stats_dict is not None:
        stats_dict["frames_total"] = N
        stats_dict["frames_voiced"] = voiced
        stats_dict["frames_twoch"] = twoch
        stats_dict["voiced_ratio"] = voiced / max(1, N)
        stats_dict["twoch_ratio"] = twoch / max(1, voiced) if voiced else 0.0
        stats_dict["snr_top1_avg_db"] = snr_top1_acc / max(1, voiced)

    return y

# ============================================================
# Leichter STFT-Wiener-Postfilter (für Perzeption; für ASR oft aus!)
# ============================================================
def mild_wiener_postfilter(y, sr, nperseg=512, noverlap=256, floor_db=-12.0):
    f, t, Y = stft(y, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Y)
    frame_energy = np.mean(mag, axis=0)
    k = max(1, int(0.2 * len(frame_energy)))
    noise_idx = np.argpartition(frame_energy, k)[:k]
    Npsd = np.mean(np.abs(Y[:, noise_idx])**2, axis=1, keepdims=True) + 1e-12
    Spsd = np.abs(Y)**2
    snr = Spsd / (Npsd + 1e-12)
    gain = snr / (snr + 1.0)
    gain = np.sqrt(np.clip(gain, 10.0**(floor_db/20.0), 1.0))
    Yf = Y * gain
    _, y_hat = istft(Yf, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=None)
    if len(y_hat) < len(y):
        y_hat = np.pad(y_hat, (0, len(y)-len(y_hat)))
    elif len(y_hat) > len(y):
        y_hat = y_hat[:len(y)]
    return y_hat.astype(np.float32)

# ============================================================
# (Optional) DoA via SRP-PHAT + Delay-and-Sum (mit Logging)
# ============================================================
def steering_delays(mic_pos, az_deg):
    az = np.deg2rad(az_deg)
    u = np.array([np.cos(az), np.sin(az)], dtype=np.float32)
    center = np.mean(mic_pos, axis=0, keepdims=True)
    rel = mic_pos - center
    proj = rel @ u
    return proj / SPEED_OF_SOUND  # Sekunden

def doa_track_and_beamform(X, sr):
    t0 = time.perf_counter()
    C, T = X.shape
    nperseg, noverlap = WF_NPERSEG, WF_NOVERLAP
    f, t_grid, Xstft = [], [], []
    for m in range(C):
        f_m, t_m, X_m = stft(X[m], fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, boundary=None)
        Xstft.append(X_m)
        if m == 0:
            f, t_grid = f_m, t_m
    F, N = Xstft[0].shape
    Xstft = np.stack(Xstft, axis=0)  # [C, F, N]
    freqs = f.astype(np.float32)
    two_pi = 2 * np.pi
    log(f"DoA: STFT F={F}, N={N}")

    delay_table = {az: steering_delays(MIC_POS, az) for az in DOA_AZIMUTH_GRID_DEG}

    az_hist, conf_hist = [], []
    last_print = time.perf_counter()
    for i in range(N):
        Xf = Xstft[:, :, i]  # [C, F]
        eps = 1e-12
        R = np.zeros((C, C, F), dtype=np.complex64)
        for m in range(C):
            for n in range(C):
                cross = Xf[m] * np.conj(Xf[n])
                R[m, n] = cross / (np.abs(cross) + eps)

        best_val = -1e18
        best_az = 0.0
        vals = []
        for az in DOA_AZIMUTH_GRID_DEG:
            tau = delay_table[az]
            val = 0.0
            for m in range(C):
                for n in range(m+1, C):
                    phase = np.exp(1j * two_pi * freqs * (tau[m] - tau[n]))
                    val += np.real(np.sum(R[m, n] * phase))
            vals.append(val)
            if val > best_val:
                best_val = val
                best_az = az
        vals = np.array(vals, dtype=np.float32)
        peak = float(best_val)
        med = float(np.median(vals))
        rng = float(np.max(vals) - np.min(vals) + 1e-12)
        conf = np.clip((peak - med) / (rng + 1e-12), 0.0, 1.0)

        if len(az_hist) == 0:
            az_s = best_az
        else:
            az_s = DOA_ALPHA_SMOOTH * az_hist[-1] + (1.0 - DOA_ALPHA_SMOOTH) * best_az

        az_hist.append(az_s)
        conf_hist.append(conf)

        if (time.perf_counter() - last_print) > 0.5:
            print(f" DoA: {i}/{N} Frames ({100*i/N:.1f}%)", end="\r")
            last_print = time.perf_counter()
    print()
    t1 = time.perf_counter()
    log(f"DoA: Winkel geschätzt in {seconds_to_str(t1 - t0)}")

    # DS-Beamforming nur bei vertrauenswürdigen Frames
    Ybf = np.zeros((F, N), dtype=np.complex64)
    valid = np.zeros(N, dtype=bool)
    for i in range(N):
        if conf_hist[i] < DOA_MIN_CONF:
            continue
        az = az_hist[i]
        tau = delay_table[min(DOA_AZIMUTH_GRID_DEG, key=lambda z: abs(z - az))]
        A = np.exp(-1j * two_pi * freqs[:, None] * tau[None, :])  # [F, C]
        steered = np.sum(A * Xstft.transpose(1,0,2)[:, :, i], axis=1) / C
        Ybf[:, i] = steered
        valid[i] = True

    _, y_doa = istft(Ybf, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=None)
    t2 = time.perf_counter()
    log(f"DoA: DS & iSTFT in {seconds_to_str(t2 - t1)}")

    doa_info = {
        "t_sec": np.array(t_grid, dtype=np.float32),
        "azimuth_smooth_deg": np.array(az_hist, dtype=np.float32),
        "confidence": np.array(conf_hist, dtype=np.float32),
        "valid_frames": valid,
    }
    return y_doa.astype(np.float32), doa_info

# ============================================================
# Main
# ============================================================
def main():
    global _t0_global
    _t0_global = time.perf_counter()
    log("Starte Pipeline")

    # 1) Dateien finden & laden
    mic_wavs = sorted(glob.glob(os.path.join(IN_DIR, GLOB_PATTERN)))
    if len(mic_wavs) != 4:
        raise RuntimeError(f"Erwarte 4 Dateien, gefunden {len(mic_wavs)}: {mic_wavs}")
    log("Verwende Mikrofone:\n  - " + "\n  - ".join(mic_wavs))

    xs, srs = [], []
    for p in mic_wavs:
        x, sr = sf.read(p, always_2d=False)
        xs.append(ensure_mono(x))
        srs.append(sr)
    if len(set(srs)) != 1:
        raise ValueError(f"Samplerates uneinheitlich: {srs}")
    sr = srs[0]
    log(f"Samplerate: {sr} Hz")

    # 2) Optional kürzen
    if MAX_SECONDS is not None:
        max_len = int(MAX_SECONDS * sr)
        xs = [x[:max_len] for x in xs]
        log(f"Zeitbegrenzung aktiv: {MAX_SECONDS}s")

    # 3) Stapeln & Normieren
    minlen = min(len(x) for x in xs)
    X = np.stack([x[:minlen] for x in xs], axis=0)
    peak = np.max(np.abs(X), axis=1, keepdims=True) + 1e-12
    X = X / peak
    dur_sec = minlen / sr

    frame_len = int(sr * FRAME_MS / 1000.0)
    hop = int(sr * HOP_MS / 1000.0)
    N_frames = 1 + max(0, (minlen - frame_len) // hop)
    log(f"Audiodauer ≈ {dur_sec:.1f}s, Samples={minlen}, Frames={N_frames} (frame={frame_len}, hop={hop})")

    timings = {}

    # 4) SNR-Gating (ASR-mode: nur 1 Kanal)
    t1 = time.perf_counter()
    last_print = time.perf_counter()
    def progress(i, N):
        nonlocal last_print
        now = time.perf_counter()
        if now - last_print > 0.5:
            print(f" SNR-Gating: {i}/{N} Frames ({100*i/N:.1f}%)", end="\r")
            last_print = now

    gating_stats = {}
    log("SNR-Gating startet …")
    y_snr = snr_gating_mixer(
        X, sr,
        frame_ms=FRAME_MS, hop_ms=HOP_MS,
        min_dbfs=MIN_DBFS, min_snr_db=MIN_SNR_DB,
        snr_within_db=SNR_WITHIN_DB, release_frames=RELEASE_FR,
        allow_two_channels=(not ASR_MODE),
        stats_dict=gating_stats,
        progress_callback=progress
    )
    print()
    t2 = time.perf_counter()
    timings["snr_gating_sec"] = t2 - t1
    log(f"SNR-Gating fertig in {seconds_to_str(timings['snr_gating_sec'])} "
        f"(voiced={gating_stats.get('voiced_ratio',0):.2f}, twoch={gating_stats.get('twoch_ratio',0):.2f}, "
        f"top1_snr_avg={gating_stats.get('snr_top1_avg_db',0):.1f} dB)")

    # 5) Optional: DoA + DS (für WER erst aktivieren, wenn MIC_POS/Reihenfolge 100% sicher ist)
    if USE_DOA_DS:
        log("DoA+DS aktiviert …")
        t3 = time.perf_counter()
        y_doa, doa_info = doa_track_and_beamform(X, sr)
        t4 = time.perf_counter()
        timings["doa_total_sec"] = t4 - t3
        conf_med = float(np.median(doa_info["confidence"])) if len(doa_info["confidence"]) else 0.0
        valid_pct = 100.0 * np.mean(doa_info["valid_frames"]) if len(doa_info["valid_frames"]) else 0.0
        log(f"DoA-Pfad fertig in {seconds_to_str(timings['doa_total_sec'])} "
            f"(median conf={conf_med:.2f}, gültige DS-Frames={valid_pct:.1f}%)")
        y_frontend = y_doa
    else:
        y_frontend = y_snr

    # 6) Postfilter/HPF/Normierung
    log("Postprocessing …")
    t5 = time.perf_counter()
    y = butter_highpass(y_frontend, sr, fc=HPF_HZ, order=2)
    if not ASR_MODE:
        y = mild_wiener_postfilter(y, sr, nperseg=WF_NPERSEG, noverlap=WF_NOVERLAP, floor_db=WF_FLOOR_DB)
    y = peak_normalize(y, target_peak=TARGET_PEAK)
    t6 = time.perf_counter()
    timings["post_sec"] = t6 - t5
    log(f"Post fertig in {seconds_to_str(timings['post_sec'])} (ASR_MODE={'ON' if ASR_MODE else 'OFF'})")

    # 7) Outputs
    out_base = os.path.join(IN_DIR, "out")
    os.makedirs(out_base, exist_ok=True)
    # a) Frontend roh (für WER)
    out_front = os.path.join(out_base, "frontend.wav")
    sf.write(out_front, y_frontend, sr)
    # b) Endsignal (mit HPF, ggf. Postfilter)
    tag = ("asr" if ASR_MODE else ("doa" if USE_DOA_DS else "snr"))
    out_path = os.path.join(out_base, f"final_{tag}.wav")
    sf.write(out_path, y, sr)
    log(f"Outputs: \n  - Frontend roh: {out_front}\n  - Final:        {out_path}")

    # 8) Metriken
    total = time.perf_counter() - _t0_global
    realtime_factor = (dur_sec / total) if total > 0 else float("nan")
    timings["total_sec"] = total
    timings["audio_sec"] = dur_sec
    timings["rt_factor"] = realtime_factor

    print("\n==================== Laufzeit-Metriken ====================")
    print(f"Gesamtzeit:       {seconds_to_str(total)}")
    print(f"Audio-Dauer:      {seconds_to_str(dur_sec)}")
    print(f"SNR-Gating:       {seconds_to_str(timings['snr_gating_sec'])}")
    if USE_DOA_DS and 'doa_total_sec' in timings:
        print(f"DoA+DS gesamt:    {seconds_to_str(timings['doa_total_sec'])}")
    print(f"Post:             {seconds_to_str(timings['post_sec'])}")
    print(f"Echtzeit-Faktor:  {realtime_factor:.2f}x ( >1 = schneller als Echtzeit )")
    print(f"Voiced-Ratio:     {gating_stats.get('voiced_ratio',0):.2f}")
    print(f"Two-Channel-Ratio:{gating_stats.get('twoch_ratio',0):.2f} (ASR_MODE={ASR_MODE})")
    print("==========================================================\n")

    metrics_path = os.path.join(out_base, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("# Laufzeit- & Frontend-Metriken\n")
        for k, v in timings.items():
            f.write(f"{k}={v}\n")
        for k, v in gating_stats.items():
            f.write(f"{k}={v}\n")
        f.write(f"frame_len={int(sr*FRAME_MS/1000.0)}\n")
        f.write(f"hop={int(sr*HOP_MS/1000.0)}\n")
        f.write(f"sr={sr}\n")
        f.write(f"channels=4\n")
        f.write("files=\n")
        for w in mic_wavs:
            f.write(f"  {w}\n")
    log(f"Metriken gespeichert: {metrics_path}")

if __name__ == "__main__":
    main()
