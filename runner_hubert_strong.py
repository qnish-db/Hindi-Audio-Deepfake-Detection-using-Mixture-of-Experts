# ===== HuBERT-Base — subset runner (resumable, quiet, strong-only) =====
from pathlib import Path
import os, csv, time, gc, sys, signal
import numpy as np, pandas as pd
from tqdm import tqdm

# ---------- CONFIG ----------
ROOT        = Path(r"G:\My Drive\hindi_dfake")   # adjust if needed
PTM_NAME    = "hubert-base"
MODEL_ID    = "facebook/hubert-base-ls960"
TARGET_SR   = 16000
SAVE_DTYPE  = "float16"
DRY_RUN     = False
LAST_K      = int(os.environ.get("PTM_LASTK", "1"))  # pool last-K hidden layers (default 1)

# Optional cap per run (None = process all pending this session)
CAP = int(os.environ.get("PTM_CAP", "0")) or None

# ---------- Fixed paths ----------
PROC_DIR   = ROOT / "processed"
WAV_DIR    = PROC_DIR / "wav"
FEAT_DIR   = PROC_DIR / "features" / "ptm"
META_DIR   = ROOT / "metadata"
MANIFEST   = META_DIR / "features_manifest.csv"
SUBSET_V2F = META_DIR / "ptm_subset.v2.fixed.csv"   # strong-only paths (all fake + 50% real)
JOBS_DIR   = META_DIR / "ptm_jobs"
TODO_CSV   = JOBS_DIR / f"{PTM_NAME}.todo.csv"
HF_CACHE   = ROOT / "models" / "hf-cache"

for d in (FEAT_DIR, META_DIR, JOBS_DIR, HF_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---------- PATH SHIM ----------
_OLD_ROOTS = [
    "/content/drive/MyDrive/hindi_dfake",
    "C:/content/drive/MyDrive/hindi_dfake",
    "C:\\content\\drive\\MyDrive\\hindi_dfake",
    "G:/My Drive/hindi_dfake",
    "G:\\My Drive\\hindi_dfake",
]
def _norm(s: str) -> str:
    return str(s).replace("\\", "/")

def normalize_to_root(path_str: str) -> Path:
    s = _norm(path_str)
    # already under current ROOT?
    if _norm(str(ROOT)) in s:
        return Path(s)
    # map known old roots to current root
    for old in _OLD_ROOTS:
        o = _norm(old)
        if s.startswith(o):
            rel = s[len(o):].lstrip("/")
            return ROOT / Path(rel)
    # strong profile tail resolver
    needle = "/processed/wav/strong/"
    i = s.lower().find(needle)
    if i != -1:
        tail = s[i + len(needle):]
        return WAV_DIR / "strong" / Path(tail)
    # fallback: join filename (should not happen with fixed csv)
    return ROOT / Path(path_str).name

def vec_path_for(ptm_name: str, path_audio, profile: str="strong") -> Path:
    pa = Path(normalize_to_root(path_audio))
    base = (WAV_DIR / profile).resolve()
    try:
        tail = pa.resolve().relative_to(base)
    except Exception:
        s = _norm(str(pa))
        needle = f"/wav/{profile}/"
        i = s.lower().find(needle)
        tail = Path(s[i + len(needle):]) if i != -1 else Path(pa.name)
    out = FEAT_DIR / ptm_name / tail.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

# ---------- Device ----------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(1)
else:
    torch.set_float32_matmul_precision("high")
print(f"[env] DEVICE={DEVICE}")

# ---------- Sanity ----------
def sanity():
    for p in (ROOT, PROC_DIR, WAV_DIR, META_DIR):
        if not p.exists(): raise FileNotFoundError(f"Missing folder: {p}")
    if not SUBSET_V2F.exists(): raise FileNotFoundError(f"Missing subset file: {SUBSET_V2F}")
    for pkg in ("torch","numpy","pandas","soundfile","transformers"):
        __import__(pkg)
sanity()

# ---------- IO helpers ----------
def load_pcm16_mono(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != TARGET_SR: raise RuntimeError(f"{path}: expected {TARGET_SR} Hz, got {sr}")
    if x.ndim > 1: x = x.mean(axis=1)
    np.clip(x, -1.0, 1.0, out=x)
    return x

def _ensure_manifest_header():
    if not MANIFEST.exists() and not DRY_RUN:
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["path_audio","profile","ptm","vec_path","dim","seconds"])

def append_manifest_rows(rows):
    if DRY_RUN or not rows: return
    _ensure_manifest_header()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

# ---------- Load subset (strong-only) + build/refresh TODO ----------
subset = pd.read_csv(SUBSET_V2F)
subset = subset[subset["profile"]=="strong"].copy()
if subset.empty:
    raise RuntimeError("No rows with profile=strong in ptm_subset.v2.fixed.csv")

subset["path_audio"] = subset["path_audio"].map(lambda p: str(normalize_to_root(p)))

if Path(TODO_CSV).exists():
    todo = pd.read_csv(TODO_CSV)
else:
    todo = subset.copy()
    todo["vec_path"] = todo["path_audio"].map(lambda p: str(vec_path_for(PTM_NAME, p, "strong")))
    todo["status"]   = "PENDING"
    todo.to_csv(TODO_CSV, index=False)

# Re-scan: mark DONE if npy exists or manifest already has it
done_manifest = set()
if MANIFEST.exists():
    try:
        mf = pd.read_csv(MANIFEST, usecols=["ptm","vec_path"])
        done_manifest = set(mf.loc[mf["ptm"]==PTM_NAME, "vec_path"].astype(str))
    except Exception:
        pass

print("[rescan] Checking current status …")
exists = []
in_mani = []
for vp in tqdm(todo["vec_path"].astype(str).tolist(), desc="rescan", unit="file"):
    exists.append(Path(vp).exists())
    in_mani.append(vp in done_manifest)
todo.loc[np.array(exists) | np.array(in_mani), "status"] = "DONE"
todo.to_csv(TODO_CSV, index=False)

pending = todo[todo["status"]=="PENDING"]
if CAP is not None:
    pending = pending.head(CAP)
print(f"[resume] Pending now: {len(pending)} (of {len(todo)}) | DRY_RUN={DRY_RUN} | CAP={CAP} | LAST_K={LAST_K}")

# ---------- Load HF model ----------
os.environ["HF_HOME"] = str(HF_CACHE)
from transformers import AutoFeatureExtractor, AutoModel
fe  = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
m   = AutoModel.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE),
        use_safetensors=True
    ).to(DEVICE).eval()

@torch.inference_mode()
def pool_lastk(hidden_states, last_k=1):
    hs = hidden_states[-last_k:] if last_k>0 else [hidden_states[-1]]
    x  = torch.stack(hs, dim=0).mean(dim=0)  # (B,T,H)
    x  = x[0]                                # (T,H)
    mu = x.mean(0)
    sd = x.std(0, unbiased=False)
    return torch.cat([mu, sd], 0).cpu().numpy().astype("float32")

# ---------- Graceful exit (flush before exit) ----------
rows_buf = []
def _flush_and_exit(code=0):
    if not DRY_RUN and rows_buf:
        append_manifest_rows(rows_buf)
        todo.to_csv(TODO_CSV, index=False)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n[exit] Progress saved.")
    sys.exit(code)

def _sigint_handler(signum, frame):
    print("\n[ctrl-c] Caught interrupt, saving progress …")
    _flush_and_exit(0)

signal.signal(signal.SIGINT, _sigint_handler)

# ---------- Main loop ----------
FLUSH_EVERY = 200
created = 0
with tqdm(total=len(pending), desc=PTM_NAME, unit="file") as pbar:
    for idx, row in pending.iterrows():
        pa, profile, vp = row["path_audio"], row.get("profile", "strong"), row["vec_path"]
        try:
            wav = load_pcm16_mono(pa)
            batch = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt")
            out = m(batch.input_values.to(DEVICE),
                    output_hidden_states=True,
                    return_dict=True)
            hs  = out.hidden_states if getattr(out, "hidden_states", None) is not None else [out.last_hidden_state]
            vec = pool_lastk(hs, last_k=LAST_K)

            if not DRY_RUN:
                np.save(vp, vec.astype(SAVE_DTYPE))
                dur = len(wav)/TARGET_SR
                rows_buf.append([pa, "strong", PTM_NAME, vp, int(vec.shape[0]), round(dur,3)])
                todo.loc[row.name, "status"] = "DONE"
            created += 1

            if created % FLUSH_EVERY == 0 and not DRY_RUN:
                append_manifest_rows(rows_buf); rows_buf.clear()
                todo.to_csv(TODO_CSV, index=False)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"\n   [warn] {pa}: {e}")

        finally:
            pbar.update(1)

# final flush
_flush_and_exit(0)
