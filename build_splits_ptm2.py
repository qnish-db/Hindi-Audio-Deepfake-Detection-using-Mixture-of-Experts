# build_splits_ptm2.py  (balanced by FILE COUNTS, still speaker-disjoint)
from pathlib import Path
import os, csv, random, hashlib
import pandas as pd

# ---------------- CONFIG ----------------
ROOT = Path(r"G:\My Drive\hindi_dfake")
META_DIR = ROOT / "metadata"
FEAT_MANIFEST = META_DIR / "features_manifest.csv"

PTMS = ["wav2vec2-base", "hubert-base"]
PROFILE = "strong"

TRAIN_CSV = META_DIR / "split_train.ptm2.csv"
VAL_CSV   = META_DIR / "split_val.ptm2.csv"
TEST_CSV  = META_DIR / "split_test.ptm2.csv"

# target file ratios per label (apply to #FILES, not #speakers)
RATIOS = (0.80, 0.10, 0.10)  # train, val, test
SEED = 1337
random.seed(SEED)

# master column candidates
CAND_PATH_COLS = ["path_audio", "path", "wav_path", "audio_path"]
CAND_UID_COLS  = ["utt_id", "id", "uid"]
CAND_SPK_COLS  = ["speaker_id", "spk", "speaker"]
CAND_LABEL_COL = "label"
# ----------------------------------------


def _norm(s: str) -> str:
    return str(s).replace("\\", "/")


def _pick_col(df, choices, required=False, fallback=None):
    for c in choices:
        if c in df.columns:
            return c
    if required and fallback is None:
        raise KeyError(f"Missing any of columns: {choices}")
    return fallback


def _map_raw_to_strong(path_str: str) -> str:
    s = _norm(path_str)
    if "/processed/wav/strong/" in s:
        return s
    if "/raw/" in s:
        tail = s.split("/raw/", 1)[1]
        return f"{_norm(ROOT)}/processed/wav/strong/raw/{tail}"
    base = os.path.basename(s)
    return f"{_norm(ROOT)}/processed/wav/strong/raw/{base}"


def _bucket_from_stem(path_str: str, K=128) -> str:
    stem = os.path.splitext(os.path.basename(path_str))[0]
    h = int(hashlib.sha1(stem.encode("utf-8")).hexdigest(), 16)
    return f"fakebucket_{h % K}"


def _read_master(p: Path, label_value: int) -> pd.DataFrame:
    df = pd.read_csv(p)
    path_col = _pick_col(df, CAND_PATH_COLS, required=True)
    df["__path_audio"] = df[path_col].astype(str).map(_map_raw_to_strong)

    if CAND_LABEL_COL not in df.columns:
        df[CAND_LABEL_COL] = label_value
    else:
        df[CAND_LABEL_COL] = df[CAND_LABEL_COL].map(
            lambda x: 1 if str(x).lower() in {"1", "fake", "tts", "spoof"} else 0
        )

    uid_col = _pick_col(df, CAND_UID_COLS, required=False, fallback=None)
    spk_col = _pick_col(df, CAND_SPK_COLS, required=False, fallback=None)

    # Better "speakers" for fakes
    if label_value == 1:
        voice = df["voice"] if "voice" in df.columns else None
        tts   = df["tts_model"] if "tts_model" in df.columns else None
        if voice is not None or tts is not None:
            df["speaker_id"] = (
                (voice.fillna("voice?") if voice is not None else "voice?").astype(str)
                + "|" +
                (tts.fillna("tts?") if tts is not None else "tts?").astype(str)
            )
            spk_col = "speaker_id"
        else:
            df["speaker_id"] = df["__path_audio"].map(lambda p: _bucket_from_stem(p, K=128))
            spk_col = "speaker_id"

    # Normalize column names
    if uid_col and "utt_id" not in df.columns:
        df = df.rename(columns={uid_col: "utt_id"})
    if "utt_id" not in df.columns:
        # ensure unique-ish id
        df["utt_id"] = [f"{label_value}_{i}" for i in range(len(df))]

    if spk_col and spk_col != "speaker_id":
        df = df.rename(columns={spk_col: "speaker_id"})
    if "speaker_id" not in df.columns:
        df["speaker_id"] = f"unknown_{label_value}"

    return df[["utt_id", "__path_audio", "speaker_id", CAND_LABEL_COL]].copy()


def _load_masters():
    mr = META_DIR / "master_real.csv"
    mf = META_DIR / "master_fake.csv"
    if not mr.exists() or not mf.exists():
        raise FileNotFoundError("master_real.csv or master_fake.csv not found in metadata/")
    dfr = _read_master(mr, 0)
    dff = _read_master(mf, 1)
    all_rows = pd.concat([dfr, dff], ignore_index=True)
    all_rows["__path_audio"] = all_rows["__path_audio"].astype(str).map(_norm)
    return all_rows


def _load_feat_manifest():
    if not FEAT_MANIFEST.exists():
        raise FileNotFoundError(f"{FEAT_MANIFEST} not found.")
    mf = pd.read_csv(FEAT_MANIFEST)
    need = ["path_audio", "ptm", "vec_path", "profile"]
    missing = [c for c in need if c not in mf.columns]
    if missing:
        raise KeyError(f"features_manifest.csv missing columns: {missing}")
    mf = mf[(mf["profile"] == PROFILE) & (mf["ptm"].isin(PTMS))].copy()
    mf["path_audio"] = mf["path_audio"].astype(str).map(_norm)
    mf["vec_path"]   = mf["vec_path"].astype(str).map(_norm)
    return mf


def _pivot_two_ptm(mf: pd.DataFrame) -> pd.DataFrame:
    wide = mf.pivot_table(index="path_audio", columns="ptm", values="vec_path", aggfunc="first")
    for p in PTMS:
        if p not in wide.columns:
            wide[p] = None
    wide = wide.dropna(subset=PTMS, how="any").reset_index()
    rename = {p: f"vec_{p}" for p in PTMS}
    wide = wide.rename(columns=rename)
    return wide


def _greedy_balance_by_files(rows: pd.DataFrame, label_val: int, ratios=(0.8, 0.1, 0.1)):
    """
    Greedy speaker assignment that balances FILE COUNTS for a given label (0 or 1).
    Keeps speakers disjoint across splits.
    """
    df_lbl = rows[rows[CAND_LABEL_COL] == label_val].copy()
    if len(df_lbl) == 0:
        return df_lbl.iloc[0:0], df_lbl.iloc[0:0], df_lbl.iloc[0:0]

    # group by speaker -> list of rows
    g = df_lbl.groupby("speaker_id")
    spk_blocks = [grp for _, grp in g]
    # shuffle deterministically but process larger speakers first
    spk_blocks.sort(key=lambda d: (-len(d), hashlib.md5(d["speaker_id"].iloc[0].encode()).hexdigest()))

    total = len(df_lbl)
    tgt_train = int(round(total * ratios[0]))
    tgt_val   = int(round(total * ratios[1]))
    tgt_test  = total - tgt_train - tgt_val

    counts = {"train": 0, "val": 0, "test": 0}
    buckets = {"train": [], "val": [], "test": []}

    def deficit(split):
        # want to fill towards target
        t = tgt_train if split == "train" else (tgt_val if split == "val" else tgt_test)
        return t - counts[split]

    for block in spk_blocks:
        # choose split with largest positive deficit; if all negative, choose the least overfilled
        deficits = {s: deficit(s) for s in ["train", "val", "test"]}
        # pick split with max deficit (ties broken by train>val>test priority)
        split = max(deficits.items(), key=lambda kv: (kv[1], 1 if kv[0]=="train" else (0 if kv[0]=="val" else -1)))[0]
        buckets[split].append(block)
        counts[split] += len(block)

    tr = pd.concat(buckets["train"], ignore_index=True) if buckets["train"] else df_lbl.iloc[0:0]
    va = pd.concat(buckets["val"], ignore_index=True) if buckets["val"] else df_lbl.iloc[0:0]
    te = pd.concat(buckets["test"], ignore_index=True) if buckets["test"] else df_lbl.iloc[0:0]
    return tr, va, te


def _speaker_disjoint_balanced(rows: pd.DataFrame, ratios=(0.8, 0.1, 0.1)):
    # balance per label separately, then union
    tr0, va0, te0 = _greedy_balance_by_files(rows, label_val=0, ratios=ratios)  # real
    tr1, va1, te1 = _greedy_balance_by_files(rows, label_val=1, ratios=ratios)  # fake

    train = pd.concat([tr0, tr1], ignore_index=True)
    val   = pd.concat([va0, va1], ignore_index=True)
    test  = pd.concat([te0, te1], ignore_index=True)

    # shuffle deterministically
    train = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val   = val.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    test  = test.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return train, val, test


def _summarize(name, df):
    n = len(df)
    r = int((df[CAND_LABEL_COL] == 0).sum())
    f = int((df[CAND_LABEL_COL] == 1).sum())
    spk = df["speaker_id"].astype(str).nunique()
    print(f" {name:5s}: n={n} (real={r}, fake={f}), speakers={spk}")


def main():
    META_DIR.mkdir(parents=True, exist_ok=True)

    # 1) masters
    df_all = _load_masters()

    # 2) features (only strong + both PTMs)
    mf = _load_feat_manifest()
    wide = _pivot_two_ptm(mf)
    wide = wide.rename(columns={"path_audio": "__path_audio"})

    # 3) join -> keep only rows with both PTMs
    merged = df_all.merge(wide, on="__path_audio", how="inner").copy()

    # 4) finalize cols
    keep_cols = [
        "utt_id", "__path_audio", "speaker_id", CAND_LABEL_COL,
        f"vec_{PTMS[0]}", f"vec_{PTMS[1]}"
    ]
    merged = merged[keep_cols].drop_duplicates(subset=["utt_id", "__path_audio"]).copy()
    merged = merged.rename(columns={"__path_audio": "path_audio"})

    # 5) split (speaker-disjoint, file-balanced per label)
    train, val, test = _speaker_disjoint_balanced(merged, ratios=RATIOS)

    # 6) write (overwrite)
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        p.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    print("\n[done] wrote:")
    print(f"  {TRAIN_CSV}")
    print(f"  {VAL_CSV}")
    print(f"  {TEST_CSV}")

    print("\n[summary]")
    _summarize("train", train)
    _summarize("val",   val)
    _summarize("test",  test)

    # quick .npy spot-checks
    import os
    def spot(df, k=5):
        samp = df.sample(min(k, len(df)), random_state=SEED)
        for _, r in samp.iterrows():
            for p in PTMS:
                vp = r[f"vec_{p}"]
                if not os.path.exists(vp):
                    print(f"  [warn] missing vec: {vp}")
                    return False
        return True

    ok = all([spot(train), spot(val), spot(test)])
    print("\n[check] All referenced PTM .npy files exist (spot-checked)." if ok
          else "\n[check] Missing PTM .npy encountered in spot-check.")

if __name__ == "__main__":
    main()
