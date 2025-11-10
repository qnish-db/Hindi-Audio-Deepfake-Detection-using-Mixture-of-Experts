# build_test_mms_match_val_ptm2_v3.py
# Root-agnostic (joins on tail after /raw/), exact-match to VAL counts.

from pathlib import Path
import hashlib, numpy as np, pandas as pd

# ===== EDIT HERE =====
ROOT        = Path(r"G:\My Drive\hindi_dfake")
PROFILE     = "strong"
PTMS        = ["wav2vec2-base","hubert-base"]
SEED        = 2025
DISJOINT_REAL = True
DISJOINT_FAKE = False   # MMS voices overlap; keep False unless you insist
# =====================

META      = ROOT/"metadata"
MANIFEST  = META/"features_manifest.csv"
REAL_CSV  = META/"master_real.csv"
FAKE_CSV  = META/"master_fake.csv"
TRAIN_CSV = META/"split_train.ptm2.csv"
VAL_CSV   = META/"split_val.ptm2.csv"
TESTS_DIR = META/"tests"; TESTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV   = TESTS_DIR/f"test_mms.{PROFILE}.ptm2.csv"

PROC_WAV  = ROOT/"processed"/"wav"/PROFILE

def _norm(s:str)->str: return str(s).replace("\\","/")

def _tail_after_raw(path_str:str)->str:
    s = _norm(path_str)
    if "/raw/" in s: return s.split("/raw/",1)[1]
    # if it's already under processed/wav/<profile>/..., recover tail
    needle = f"/processed/wav/{PROFILE}/"
    i = s.lower().find(needle)
    if i != -1: return s[i+len(needle):]
    return Path(s).name  # fallback

def _stable_pick(df:pd.DataFrame, k:int, key:str, seed:int)->pd.DataFrame:
    if len(df)<=k: return df.copy()
    def h(x): return int(hashlib.sha1((str(x)+str(seed)).encode()).hexdigest()[:8],16)
    order = df[key].map(h).values
    return df.iloc[np.argsort(order)[:k]].copy()

def _exists(p): 
    try: return Path(p).exists()
    except: return False

# ---- targets from VAL ----
if not VAL_CSV.exists(): raise FileNotFoundError(VAL_CSV)
val = pd.read_csv(VAL_CSV)
for c in ["label","speaker_id","utt_id"]:
    if c not in val.columns: raise KeyError(f"{VAL_CSV} missing {c}")
VAL_N      = len(val)
VAL_REAL_N = int((val["label"]==0).sum())
VAL_FAKE_N = int((val["label"]==1).sum())
print(f"[target] total={VAL_N:,} | real={VAL_REAL_N:,} | fake={VAL_FAKE_N:,}")

# ---- speaker blocks from train/val (label-wise) ----
def _read_spk(csv:Path, lab:int)->set:
    if not csv.exists(): return set()
    df = pd.read_csv(csv)
    if "speaker_id" not in df.columns or "label" not in df.columns: return set()
    return set(df.loc[df["label"]==lab,"speaker_id"].astype(str))

spk_real_block = (_read_spk(TRAIN_CSV,0)|_read_spk(VAL_CSV,0)) if DISJOINT_REAL else set()
spk_fake_block = (_read_spk(TRAIN_CSV,1)|_read_spk(VAL_CSV,1)) if DISJOINT_FAKE else set()
print(f"[exclude] real_speakers={len(spk_real_block)} | fake_speakers={len(spk_fake_block)} (fake disjoint={DISJOINT_FAKE})")

# ---- load features manifest and require both PTMs on disk ----
for p in [MANIFEST, REAL_CSV, FAKE_CSV]:
    if not p.exists(): raise FileNotFoundError(p)

mf = pd.read_csv(MANIFEST)
need = ["path_audio","ptm","vec_path","profile"]
miss = [c for c in need if c not in mf.columns]
if miss: raise KeyError(f"features_manifest.csv missing columns: {miss}")

mf = mf[(mf["profile"]==PROFILE) & (mf["ptm"].isin(PTMS))].copy()
mf["path_audio"] = mf["path_audio"].astype(str).map(_norm)
mf["vec_path"]   = mf["vec_path"].astype(str).map(_norm)

wide = mf.pivot_table(index="path_audio", columns="ptm", values="vec_path", aggfunc="first").reset_index()
# ensure columns
for p in PTMS:
    if p not in wide.columns: wide[p] = None
wide = wide.rename(columns={p:f"vec_{p}" for p in PTMS})
# keep only rows with both vecs physically present
for c in [f"vec_{p}" for p in PTMS]: wide[c] = wide[c].astype(str)
wide = wide[wide.apply(lambda r: all(_exists(r[f"vec_{p}"]) for p in PTMS), axis=1)].copy()
if wide.empty: raise RuntimeError("No rows with both PTM vectors on disk.")

# derive root-agnostic tails and raw-path tails
wide["tail_proc"] = wide["path_audio"].map(_tail_after_raw)   # tail under processed
wide["tail_raw"]  = wide["tail_proc"]                         # same tail works under raw/

# ---- build master lookups by tail_raw (root-agnostic) ----
real = pd.read_csv(REAL_CSV); fake = pd.read_csv(FAKE_CSV)
for df in (real,fake):
    if "path" not in df.columns: raise KeyError("master CSV missing 'path'")
    df["path"] = df["path"].astype(str).map(_norm)
    df["tail_raw"] = df["path"].map(_tail_after_raw)

cols_r = ["utt_id","speaker_id","label","tail_raw"]
cols_f = ["utt_id","speaker_id","label","voice","tts_model","fake_type","tail_raw"]
for c in cols_r: 
    if c not in real.columns: real[c] = ""
for c in cols_f:
    if c not in fake.columns: fake[c] = ""

r_idx = real.drop_duplicates("tail_raw").set_index("tail_raw")
f_idx = fake.drop_duplicates("tail_raw").set_index("tail_raw")

def lab_of_tail(t):
    if t in r_idx.index: return 0
    if t in f_idx.index: return 1
    return None

def spk_of_tail(t):
    if t in r_idx.index:
        s = str(r_idx.loc[t,"speaker_id"]); return s if s else "unknown_0"
    if t in f_idx.index:
        row = f_idx.loc[t]
        voice = str(row.get("voice","")) or "voice?"
        tts   = str(row.get("tts_model","")) or "tts?"
        return f"{voice}|{tts}"
    return "unknown_?"

def fake_type_of_tail(t):
    return (str(f_idx.loc[t,"fake_type"]).lower() if t in f_idx.index else "")

def uid_of_tail(t):
    if t in r_idx.index: return str(r_idx.loc[t,"utt_id"])
    if t in f_idx.index: return str(f_idx.loc[t,"utt_id"])
    return Path(t).stem

wide["label"]      = wide["tail_raw"].map(lab_of_tail)
wide["speaker_id"] = wide["tail_raw"].map(spk_of_tail)
wide["fake_type"]  = wide["tail_raw"].map(fake_type_of_tail)
wide["utt_id"]     = wide["tail_raw"].map(uid_of_tail)
wide = wide[wide["label"].isin([0,1])].copy()

# ---- pools ----
cand_fake_mms = wide[(wide["label"]==1) & (wide["fake_type"]=="tts_mms")].copy()
cand_real     = wide[wide["label"]==0].copy()

if DISJOINT_FAKE and len(spk_fake_block):
    cand_fake_mms = cand_fake_mms[~cand_fake_mms["speaker_id"].astype(str).isin(spk_fake_block)]
if DISJOINT_REAL and len(spk_real_block):
    cand_real     = cand_real[~cand_real["speaker_id"].astype(str).isin(spk_real_block)]

print(f"[candidates] mms_fake={len(cand_fake_mms):,} | real={len(cand_real):,}")

# if either pool is empty, print a tiny debug sample of tails to help
if len(cand_fake_mms)==0 or len(cand_real)==0:
    print("[debug] sample tails (features) e.g.:", wide["tail_raw"].head(3).tolist())
    print("[debug] master_fake tails example:", fake["tail_raw"].head(3).tolist())
    print("[debug] master_real tails example:", real["tail_raw"].head(3).tolist())

# deterministic picks -> target counts; backfill to enforce exact numbers
def enforce_count(pool:pd.DataFrame, target:int, label_name:str)->pd.DataFrame:
    pick = _stable_pick(pool, target, key="utt_id", seed=SEED)
    if len(pick) == target: return pick
    have = len(pick); need = target - have
    print(f"[backfill] {label_name}: need {need} more.")
    # relax (ignore disjoint toggles) but stay within label/type
    if label_name=="fake":
        pool2 = wide[(wide["label"]==1) & (wide["fake_type"]=="tts_mms")].copy()
    else:
        pool2 = wide[wide["label"]==0].copy()
    already = set(pick["utt_id"].astype(str))
    pool2 = pool2[~pool2["utt_id"].astype(str).isin(already)]
    extra = _stable_pick(pool2, need, key="utt_id", seed=SEED+7)
    if len(extra) < need:
        raise RuntimeError(f"Not enough candidates to satisfy target for {label_name}. "
                           f"have={len(pick)} + {len(extra)} < {target}")
    return pd.concat([pick, extra], ignore_index=True)

pick_fake = enforce_count(cand_fake_mms, VAL_FAKE_N, "fake")
pick_real = enforce_count(cand_real,     VAL_REAL_N, "real")

# ---- finalize ptm2-wide ----
def finalize(df:pd.DataFrame)->pd.DataFrame:
    keep = ["utt_id","speaker_id","label","path_audio"] + [f"vec_{p}" for p in PTMS]
    out = df[keep].copy()
    return out

test_df = pd.concat([finalize(pick_fake), finalize(pick_real)], ignore_index=True)

# deterministic shuffle
def H(u): return int(hashlib.sha1((str(u)+str(SEED)).encode()).hexdigest()[:8],16)
test_df = test_df.iloc[np.argsort(test_df["utt_id"].map(H).values)].reset_index(drop=True)

# ensure vec columns exist
for c in [f"vec_{p}" for p in PTMS]:
    if c not in test_df.columns: test_df[c] = ""

# final hard assertion
r = int((test_df["label"]==0).sum()); f = int((test_df["label"]==1).sum())
assert len(test_df)==VAL_N and r==VAL_REAL_N and f==VAL_FAKE_N, \
    f"mismatch rows={len(test_df)} want={VAL_N} real={r}/{VAL_REAL_N} fake={f}/{VAL_FAKE_N}"

# save
test_df.to_csv(OUT_CSV, index=False)
print(f"[write] {OUT_CSV}  rows={len(test_df):,}")
print(f"[summary] real={r:,} fake={f:,} speakers={test_df['speaker_id'].astype(str).nunique():,}")
print("\n[spot] first 3 rows:\n", test_df.head(3))
