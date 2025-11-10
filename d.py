# save as: scrub_ptm_vectors.py
import argparse, os
from pathlib import Path
import pandas as pd

def detect_ptm_cols(df):
    # columns that look like npy path columns
    cols = []
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            s = df[c].dropna().astype(str)
            if not s.empty and (s.str.endswith(".npy").mean() > 0.8):
                cols.append(c)
    return cols

def path_ok(p: str, min_bytes: int) -> bool:
    try:
        st = os.stat(p)
        return st.st_size >= min_bytes
    except OSError:
        return False

def scrub_one(csv_path: Path, min_bytes: int, inplace: bool):
    df = pd.read_csv(csv_path)
    ptm_cols = detect_ptm_cols(df)
    if not ptm_cols:
        print(f"[{csv_path.name}] no .npy columns found; skipped.")
        return

    before = len(df)
    bad_counts = {c: 0 for c in ptm_cols}
    mask = []
    for _, row in df.iterrows():
        ok = True
        for c in ptm_cols:
            p = str(row.get(c, ""))
            if not p or not path_ok(p, min_bytes):
                ok = False
                bad_counts[c] += 1
                break
        mask.append(ok)

    df2 = df[pd.Series(mask, index=df.index)]
    after = len(df2)

    out = csv_path if inplace else csv_path.with_suffix(".sizeok.csv")
    df2.to_csv(out, index=False)

    print(f"[{csv_path.name}] {before} -> {after} rows (dropped {before-after}). "
          f"Checked cols: {ptm_cols} | min_bytes={min_bytes}")
    for c, nbad in bad_counts.items():
        if nbad:
            print(f"  - {c}: {nbad} rows failed")

def main():
    ap = argparse.ArgumentParser(description="Size-only scrub for PTM .npy columns in CSVs.")
    ap.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths to scrub")
    ap.add_argument("--min-bytes", type=int, default=32, help="Min file size in bytes to consider valid")
    ap.add_argument("--inplace", action="store_true", help="Overwrite original CSVs instead of writing .sizeok.csv")
    args = ap.parse_args()

    for c in args.csv:
        p = Path(c)
        if not p.exists():
            print(f"[skip] not found: {p}")
            continue
        scrub_one(p, args.min_bytes, args.inplace)

if __name__ == "__main__":
    main()
