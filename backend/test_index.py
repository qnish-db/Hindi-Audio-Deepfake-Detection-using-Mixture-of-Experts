# test_index.py
from pathlib import Path
import pandas as pd

def normalize_stem(name: str) -> str:
    # strip path + extension; lower-case; keep MMS sha tail intact
    s = (name or "").strip().replace("\\", "/").split("/")[-1]
    if "." in s:
        s = s[:s.rfind(".")]
    return s.lower()

def _choose_path_cols(df: pd.DataFrame):
    # look for any column that looks like a path to audio or vec
    cols = []
    for c in df.columns:
        if df[c].dtype == object:
            v = df[c].astype(str)
            if (v.str.contains(r"\.wav$|\.flac$|\.mp3$|\.npy$", case=False, regex=True).mean() > 0.2):
                cols.append(c)
    if cols:
        return cols
    # fallback common names
    for c in ("path","path_audio","wav","audio_path"):
        if c in df.columns:
            return [c]
    return []

def build_truth_index(csv_path: Path):
    meta = {"csv": str(csv_path), "rows": 0, "path_cols": [], "reason": ""}
    index = {}
    try:
        if not csv_path or not Path(csv_path).exists():
            meta["reason"] = "csv_missing"
            return index, meta
        df = pd.read_csv(csv_path)
        meta["rows"] = len(df)
        if "label" not in df.columns:
            meta["reason"] = "no_label_col"
            return index, meta
        path_cols = _choose_path_cols(df)
        meta["path_cols"] = path_cols
        if not path_cols:
            meta["reason"] = "no_path_like_cols"
            return index, meta
        # build by any path-like column
        for c in path_cols:
            for v, lab in zip(df[c].astype(str).fillna(""), df["label"].astype(int)):
                stem = normalize_stem(v)
                if stem and stem not in index:
                    index[stem] = int(lab)
        meta["reason"] = "ok"
        return index, meta
    except Exception as e:
        meta["reason"] = f"exception:{e}"
        return {}, meta
