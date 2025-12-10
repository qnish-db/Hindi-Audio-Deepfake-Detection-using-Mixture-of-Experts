# preprocess_strong.py
import subprocess, uuid, time, hashlib, os
from pathlib import Path
import numpy as np
import soundfile as sf

TARGET_SR   = 16000
TRIM_THR_DB = -45
TRIM_DUR_S  = 0.20
TARGET_DB   = -26.0  # per-file RMS target

def _rms_dbfs_arr(x: np.ndarray) -> float:
    if x.ndim > 1: x = x.mean(axis=1)
    if len(x) == 0: return -120.0
    
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 1e-9: return -120.0
    return 20.0*np.log10(min(max(rms, 1e-9), 1.0))

def _rng_from_key(key: str):
    seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def _add_colored_noise(x, sr, rng, band=(2500,6000), snr_db=25.0):
    n = len(x)
    if n == 0: return x
    wn = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(wn); freqs = np.fft.rfftfreq(n, d=1.0/sr)
    mask = (freqs >= band[0]) & (freqs <= band[1]); X[~mask] = 0.0
    noise = np.fft.irfft(X, n=n).astype(np.float32)
    sig_rms = np.sqrt(np.mean(np.square(x))) + 1e-9
    target_noise_rms = sig_rms / (10.0**(snr_db/20.0))
    cur_rms = np.sqrt(np.mean(np.square(noise))) + 1e-12
    noise *= (target_noise_rms / cur_rms)
    return np.clip(x + noise, -1.0, 1.0)

def _add_impulses(x, sr, rng, per_sec=1.0, gain=0.08):
    n = len(x)
    if n == 0: return x
    dur = n/float(sr); k = max(1, int(per_sec * dur))
    idx = rng.integers(0, n, size=k)
    amp = gain * (np.sqrt(np.mean(np.square(x))) + 1e-9)
    y = x.copy()
    y[idx] = np.clip(y[idx] + amp * rng.choice([-1.0, 1.0], size=k), -1.0, 1.0)
    return y

def _add_small_reverb(x, sr, rng, t_sec=0.03, decay=0.35, wet=0.18):
    n = len(x)
    if n == 0: return x
    ir_len = max(8, int(t_sec * sr))
    t = np.arange(ir_len, dtype=np.float32)
    ir = np.exp(-decay * t / ir_len).astype(np.float32)
    for _ in range(3):
        pos = int(rng.integers(0, ir_len))
        ir[pos] += float(rng.uniform(0.1, 0.3))
    ir /= (np.sum(np.abs(ir)) + 1e-9)
    y = np.convolve(x, ir, mode="full")[:n].astype(np.float32)
    return np.clip((1.0 - wet) * x + wet * y, -1.0, 1.0)

def rawboost_v3(x: np.ndarray, sr: int, key: str) -> np.ndarray:
    """
    RawBoost v3 augmentation - EXACT match to training notebook.
    
    Applies (deterministically based on key):
    - Colored noise (2500-6000 Hz, SNR=25dB) with prob=1.0
    - Impulses (1.0/sec, gain=0.08) with prob=0.5
    - Small reverb (t=0.03s, decay=0.35, wet=0.18) with prob=0.6
    """
    if x.ndim > 1: x = x.mean(axis=1)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    rng = _rng_from_key(key)
    
    # MATCH TRAINING: Apply with probabilities
    if rng.uniform() < 1.0:  # RB_NOISE_PROB = 1.0 (always)
        x = _add_colored_noise(x, sr, rng, (2500,6000), 25.0)
    if rng.uniform() < 0.5:  # RB_IMPULSE_PROB = 0.5
        x = _add_impulses(x, sr, rng, 1.0, 0.08)
    if rng.uniform() < 0.6:  # RB_REVERB_PROB = 0.6
        x = _add_small_reverb(x, sr, rng, 0.03, 0.35, 0.18)
    return x

def _ffmpeg(*args):
    return subprocess.run(
        ["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

def _build_filter_chain_strong(gain_db: float) -> str:
    return ",".join([
        "highpass=f=20",
        "equalizer=f=3000:t=q:w=1.0:g=2.5",
        "equalizer=f=4800:t=q:w=0.9:g=2.0",
        "treble=g=1.0:f=6000:t=h:w=0.7",
        f"volume={gain_db}dB",
        f"silenceremove=start_periods=1:start_duration={TRIM_DUR_S}:start_threshold={TRIM_THR_DB}dB:stop_periods=1:stop_duration={TRIM_DUR_S}:stop_threshold={TRIM_THR_DB}dB"
    ])

def preprocess_strong_from_path(path_in: Path):
    """Return (wav_float32_mono_16k, sr, debug_times) using EXACT training chain."""
    t0 = time.perf_counter()
    dbg = {}

    # Read & (if needed) resample to 16k PCM16 via ffmpeg to match training
    t = time.perf_counter()
    tmp_res = Path(path_in).with_suffix(f".res16k.{uuid.uuid4().hex}.wav")
    p = _ffmpeg("-i", str(path_in), "-ac","1","-ar", str(TARGET_SR), "-sample_fmt","s16", str(tmp_res))
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {p.stderr.decode('utf-8','ignore')}")
    x, sr = sf.read(str(tmp_res), dtype="float32", always_2d=False)
    dbg["t_read_resample_ms"] = int((time.perf_counter() - t) * 1000)

    # RawBoost v3 — deterministic by path
    t = time.perf_counter()
    y = rawboost_v3(x, TARGET_SR, f"{path_in}|strong")
    dbg["t_rawboost_ms"] = int((time.perf_counter() - t) * 1000)

    # Write RB temp, then EQ+gain+trim with ffmpeg
    tmp_rb  = Path(path_in).with_suffix(f".rb.{uuid.uuid4().hex}.wav")
    tmp_out = Path(path_in).with_suffix(f".tmpout.{uuid.uuid4().hex}.wav")
    sf.write(str(tmp_rb), y, TARGET_SR, subtype="PCM_16")

    gain_db = float(np.clip(TARGET_DB - _rms_dbfs_arr(y), -20.0, 20.0))
    filt = _build_filter_chain_strong(gain_db)

    t = time.perf_counter()
    p = _ffmpeg("-i", str(tmp_rb), "-ac","1","-ar", str(TARGET_SR), "-af", filt, "-sample_fmt","s16", str(tmp_out))
    dbg["t_eq_gain_trim_ms"] = int((time.perf_counter() - t) * 1000)
    try: tmp_rb.unlink(missing_ok=True)
    except: pass
    if p.returncode != 0:
        try: tmp_out.unlink(missing_ok=True)
        except: pass
        raise RuntimeError(f"ffmpeg filter failed: {p.stderr.decode('utf-8','ignore')}")

    z, sr2 = sf.read(str(tmp_out), dtype="float32", always_2d=False)
    try: tmp_out.unlink(missing_ok=True)
    except: pass

    dbg["t_pre_ms"] = int((time.perf_counter() - t0) * 1000)
    return z.astype(np.float32, copy=False), sr2, dbg
