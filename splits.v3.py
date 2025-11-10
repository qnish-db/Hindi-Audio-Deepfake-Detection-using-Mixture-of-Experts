# build_splits_ptm3.py
# Speaker-disjoint, file-count-balanced splits (train/val/test) with hard checks.
# Requires both PTM vectors present on disk; uses features_manifest.csv (profile=strong).

from pathlib import Path
import os, csv, hashlib, random
import numpy as np
import pandas as pd

# ======================= CONFIG =======================
ROOT        = Path(r"G:\My Drive\hindi_dfake")   # change if needed
META_DIR    = ROOT / "metadata"
FEAT_MANIF  = META_DIR / "features_manifest.csv"

PTMS        = ["wav2vec2-base", "hubert-base"]   # both must exist for each row
PROFILE     = "strong"                           # processed profile to use

# target file ratios per label (apply to FILE COUNTS, not speakers)
RATIOS      = (0.80, 0.10, 0.10)                 # train / val / test
SEED        = 1337
random.seed(SEED)
np.random.seed(SEED)

# outputs
TRAIN_CSV   = META_DIR / "split_train.ptm3.csv"
VAL_CSV     = META_DIR / "split_val.ptm3.csv"
TEST_CSV    = META_DIR / "split_test.ptm3.csv"

# candidate column names in masters
CAND_PATH_COLS = ["path_audio", "path", "wav_path", "audio_path"]
CAND_UID_COLS  = ["utt_id", "id", "uid"]
CAND_SPK_COLS  = ["speaker_id", "spk", "speaker"]
LABEL_COL      = "label"                          # 0=real, 1=fake
# ======================================================


def _norm(s: str) -> str:
    return str(s).replace("\\", "/").strip()

def _pick_col(df: pd.DataFrame, choices, required=False, fallback=None):
    for c in choices:
        if c in df.columns:
            return c
    if required and fallback is None:
        raise KeyError(f"Missing any of columns: {choices}")
    return fallback

def _map_raw_to_strong(path_str: str) -> str:
    """Root-agnostic: mirror tail under /processed/wav/<PROFILE>/raw/... if needed."""
    s = _norm(path_str).lower()
    if f"/processed/wav/{PROFILE}/" in s:
        return _norm(path_str)
    if "/raw/" in s:
        tail = s.split("/raw/", 1)[1]
        return f"{_norm(ROOT)}/processed/wav/{PROFILE}/raw/{tail}"
    base = os.path.basename(s)
    return f"{_norm(ROOT)}/processed/wav/{PROFILE}/raw/{base}"

def _bucket_from_stem(path_str: str, K=256) -> str:
    stem = os.path.splitext(os.path.basename(path_str))[0]
    h = int(hashlib.sha1(stem.encode("utf-8")).hexdigest(), 16)
    return f"fakebucket_{h % K}"

def _read_master(p: Path, default_label: int) -> pd.DataFrame:
    df = pd.read_csv(p)
    path_col = _pick_col(df, CAND_PATH_COLS, required=True)
    df["__path_audio"] = df[path_col].astype(str).map(_map_raw_to_strong)

    # label normalization -> 0/1
    if LABEL_COL not in df.columns:
        df[LABEL_COL] = default_label
    else:
        df[LABEL_COL] = df[LABEL_COL].map(
            lambda x: 1 if str(x).strip().lower() in {"1", "fake", "tts", "spoof"} else 0
        )

    uid_col = _pick_col(df, CAND_UID_COLS, required=False)
    spk_col = _pick_col(df, CAND_SPK_COLS, required=False)

    # robust fake "speaker" proxy: voice|tts_model if available; else hash bucket
    if default_label == 1:
        voice = df["voice"] if "voice" in df.columns else None
        tts   = df["tts_model"] if "tts_model" in df.columns else None
        if voice is not None or tts is not None:
            df["speaker_id"] = (
                (voice.fillna("voice?") if voice is not None else "voice?").astype(str)
                + "|" +
                (tts.fillna("tts?") if tts is not None else "tts?").astype(str)
            )
        else:
            df["speaker_id"] = df["__path_audio"].map(lambda p: _bucket_from_stem(p, K=256))
    else:
        if spk_col and spk_col != "speaker_id":
            df = df.rename(columns={spk_col: "speaker_id"})
        if "speaker_id" not in df.columns:
            df["speaker_id"] = "unknown_0"

    # stable utt_id
    if uid_col and "utt_id" not in df.columns:
        df = df.rename(columns={uid_col: "utt_id"})
    if "utt_id" not in df.columns:
        # unique-ish & deterministic
        h = df["__path_audio"].astype(str).map(lambda p: hashlib.sha1(p.encode()).hexdigest()[:8])
        df["utt_id"] = [f"{default_label}_{i}_{k}" for i, k in enumerate(h)]

    keep = ["utt_id", "__path_audio", "speaker_id", LABEL_COL]
    return df[keep].copy()

def _load_masters(meta_dir: Path) -> pd.DataFrame:
    mr = meta_dir / "master_real.csv"
    mf = meta_dir / "master_fake.csv"
    if not mr.exists() or not mf.exists():
        raise FileNotFoundError("Missing master_real.csv or master_fake.csv in metadata/")
    dfr = _read_master(mr, 0)
    dff = _read_master(mf, 1)
    all_rows = pd.concat([dfr, dff], ignore_index=True)
    # normalize paths once more, case-insensitive
    all_rows["__path_audio"] = all_rows["__path_audio"].astype(str).map(_norm)
    return all_rows

def _load_features_manifest(feat_csv: Path, profile: str, ptms: list) -> pd.DataFrame:
    if not feat_csv.exists():
        raise FileNotFoundError(f"{feat_csv} not found.")
    mf = pd.read_csv(feat_csv)

    need = ["path_audio", "ptm", "vec_path", "profile"]
    miss = [c for c in need if c not in mf.columns]
    if miss:
        raise KeyError(f"features_manifest.csv missing columns: {miss}")

    # keep only desired profile/PTMs
    mf = mf[(mf["profile"] == profile) & (mf["ptm"].isin(ptms))].copy()

    # warn on duplicates that pivot would collapse
    dups = mf.duplicated(subset=["path_audio", "ptm"], keep=False)
    if dups.any():
        print("[warn] duplicate (path_audio, ptm) rows in features_manifest — keeping first after pivot")

    mf["path_audio"] = mf["path_audio"].astype(str).map(_norm)
    mf["vec_path"]   = mf["vec_path"].astype(str).map(_norm)
    return mf

def _pivot_two_ptm(mf: pd.DataFrame, ptms: list) -> pd.DataFrame:
    wide = mf.pivot_table(index="path_audio", columns="ptm", values="vec_path", aggfunc="first").reset_index()
    # ensure columns exist
    for p in ptms:
        if p not in wide.columns:
            wide[p] = None
    wide = wide.rename(columns={p: f"vec_{p}" for p in ptms})
    # require both PTMs & actual files present
    for c in [f"vec_{p}" for p in ptms]:
        wide[c] = wide[c].astype(str)
    def _exists(x): 
        try: return Path(x).exists()
        except: return False
    mask = wide.apply(lambda r: all(_exists(r[f"vec_{p}"]) for p in ptms), axis=1)
    wide = wide[mask].copy()
    if wide.empty:
        raise RuntimeError("No rows with both PTM vectors present on disk.")
    return wide

def _greedy_balance_by_files(rows: pd.DataFrame, label_val: int, ratios=(0.8, 0.1, 0.1)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Greedy assign by speaker blocks to hit FILE COUNT targets for this label."""
    df_lbl = rows[rows[LABEL_COL] == label_val].copy()
    if len(df_lbl) == 0:
        return df_lbl.iloc[0:0], df_lbl.iloc[0:0], df_lbl.iloc[0:0]

    # group speakers -> blocks
    spk_blocks = [grp for _, grp in df_lbl.groupby("speaker_id")]
    # deterministic: larger first, tie-breaker by md5 of speaker_id
    spk_blocks.sort(key=lambda d: (-len(d), hashlib.md5(str(d["speaker_id"].iloc[0]).encode()).hexdigest()))

    total = len(df_lbl)
    tgt_train = int(round(total * ratios[0]))
    tgt_val   = int(round(total * ratios[1]))
    tgt_test  = total - tgt_train - tgt_val

    counts  = {"train": 0, "val": 0, "test": 0}
    target  = {"train": tgt_train, "val": tgt_val, "test": tgt_test}
    buckets = {"train": [], "val": [], "test": []}

    def deficit(split): return target[split] - counts[split]

    for block in spk_blocks:
        # pick the split with largest deficit (pref: train > val > test for ties)
        splits = ["train", "val", "test"]
        split = max(splits, key=lambda s: (deficit(s), 1 if s=="train" else (0 if s=="val" else -1)))
        buckets[split].append(block)
        counts[split] += len(block)

    tr = pd.concat(buckets["train"], ignore_index=True) if buckets["train"] else df_lbl.iloc[0:0]
    va = pd.concat(buckets["val"],   ignore_index=True) if buckets["val"]   else df_lbl.iloc[0:0]
    te = pd.concat(buckets["test"],  ignore_index=True) if buckets["test"]  else df_lbl.iloc[0:0]
    return tr, va, te

def _speaker_disjoint_balanced(rows: pd.DataFrame, ratios=(0.8, 0.1, 0.1)):
    tr0, va0, te0 = _greedy_balance_by_files(rows, label_val=0, ratios=ratios)
    tr1, va1, te1 = _greedy_balance_by_files(rows, label_val=1, ratios=ratios)
    train = pd.concat([tr0, tr1], ignore_index=True)
    val   = pd.concat([va0, va1], ignore_index=True)
    test  = pd.concat([te0, te1], ignore_index=True)
    # deterministic shuffle
    def H(u): return int(hashlib.sha1(str(u).encode()).hexdigest()[:8], 16)
    for df in (train, val, test):
        df["_h"] = df["utt_id"].map(H)
    train = train.sort_values("_h").drop(columns="_h").reset_index(drop=True)
    val   = val.sort_values("_h").drop(columns="_h").reset_index(drop=True)
    test  = test.sort_values("_h").drop(columns="_h").reset_index(drop=True)
    return train, val, test

def _summarize(name, df):
    n  = len(df)
    r  = int((df[LABEL_COL] == 0).sum())
    f  = int((df[LABEL_COL] == 1).sum())
    sp = df["speaker_id"].astype(str).nunique()
    def per_label(df_, lab):
        g = df_[df_[LABEL_COL]==lab].groupby("speaker_id").size()
        if g.empty: return (0, 0.0, 0.0)
        return int(g.sum()), float(g.mean()), float(g.median())
    r_files, r_mean, r_med = per_label(df, 0)
    f_files, f_mean, f_med = per_label(df, 1)
    print(f" {name:5s} | n={n:6d} (real={r:6d}, fake={f:6d}) | speakers={sp:5d} "
          f"| real/files spk μ/med={r_mean:.1f}/{r_med:.1f} | fake/files spk μ/med={f_mean:.1f}/{f_med:.1f}")

def _verify_disjoint(*dfs):
    # speakers
    spk_sets = [set(d["speaker_id"].astype(str)) for d in dfs]
    for i in range(len(spk_sets)):
        for j in range(i+1, len(spk_sets)):
            inter = spk_sets[i] & spk_sets[j]
            if inter:
                raise AssertionError(f"Speaker leak between splits ({i},{j}): samples={list(inter)[:5]}")
    # utt ids
    uid_sets = [set(d["utt_id"].astype(str)) for d in dfs]
    for i in range(len(uid_sets)):
        for j in range(i+1, len(uid_sets)):
            inter = uid_sets[i] & uid_sets[j]
            if inter:
                raise AssertionError(f"utt_id leak between splits ({i},{j}): samples={list(inter)[:5]}")

def main():
    META_DIR.mkdir(parents=True, exist_ok=True)

    # 1) load masters (normalize → strong paths)
    df_master = _load_masters(META_DIR)

    # 2) load features manifest (profile=PROFILE, both PTMs), require both vec files
    mf  = _load_features_manifest(FEAT_MANIF, PROFILE, PTMS)
    wide = _pivot_two_ptm(mf, PTMS)  # columns: path_audio, vec_<ptm>...

    # 3) join → keep only rows present in master
    merged = df_master.merge(wide, left_on="__path_audio", right_on="path_audio", how="inner").copy()

    # 4) tighten columns and de-dup per (utt_id, path_audio)
    keep_cols = ["utt_id", "__path_audio", "speaker_id", LABEL_COL] + [f"vec_{p}" for p in PTMS]
    merged = merged[keep_cols].rename(columns={"__path_audio": "path_audio"}).drop_duplicates(subset=["utt_id","path_audio"]).copy()

    # 5) assert vectors exist for every row (hard fail if not)
    for pcol in [f"vec_{p}" for p in PTMS]:
        missing_mask = ~merged[pcol].astype(str).map(lambda s: Path(s).exists())
        if missing_mask.any():
            bad = merged.loc[missing_mask, pcol].head(8).tolist()
            raise FileNotFoundError(f"Missing vectors in '{pcol}' (sample): {bad}")

    # 6) split (speaker-disjoint, file-count balanced per label)
    train, val, test = _speaker_disjoint_balanced(merged, ratios=RATIOS)

    # 7) leak checks
    _verify_disjoint(train, val, test)

    # 8) write
    for out in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        out.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    # 9) report
    print("\n[done] wrote:")
    print(f"  {TRAIN_CSV}")
    print(f"  {VAL_CSV}")
    print(f"  {TEST_CSV}\n")

    print("[summary]")
    _summarize("train", train)
    _summarize("val",   val)
    _summarize("test",  test)

    # 10) optional drift note vs targets (file-count per label)
    def drift(df, name):
        for lab, tag in [(0,"real"), (1,"fake")]:
            n_lab_total = int((merged[LABEL_COL]==lab).sum())
            tgt = (np.array(RATIOS) * n_lab_total).round().astype(int)
            # recompute to ensure sums align
            tgt[2] = n_lab_total - tgt[0] - tgt[1]
        # print once at end
    print("\n[checks] speaker & utt_id disjoint ✅  | both PTM vecs present ✅")

if __name__ == "__main__":
    main()
