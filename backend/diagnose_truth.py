# list_reals.py — print REAL (=0) examples from a test .csv
import argparse, pandas as pd, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to test CSV (e.g. .../test_mms.strong.ptm2.csv)")
    ap.add_argument("--n", type=int, default=30, help="How many examples to show")
    ap.add_argument("--contains", type=str, default="/raw/real_clean/", help="Substring filter on path_audio")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns or "path_audio" not in df.columns:
        raise SystemExit("[err] CSV must have columns: label, path_audio")

    reals = df[(df["label"] == 0)]
    if args.contains:
        reals = reals[reals["path_audio"].astype(str).str.contains(args.contains, case=False, na=False)]

    print(f"[csv] {args.csv}")
    print(f"[total rows] {len(df)}  |  [reals] {len(reals)}  (filter: '{args.contains}')\n")

    if reals.empty:
        print("[warn] No REAL rows matched. Try removing --contains or check the CSV path.")
        return

    reals = reals.reset_index(drop=True).head(args.n)
    for i, row in reals.iterrows():
        p = str(row["path_audio"]).replace("\\","/")
        stem = Path(p).stem
        print(f"{i+1:02d}. STEM: {stem}  |  PATH: {p}")

if __name__ == "__main__":
    main()
