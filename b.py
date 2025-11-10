# audit_ptm_coverage.py
import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np

PTMS = ["wav2vec2-base", "hubert-base"]
PROFILE = "strong"
EXPECT_DIM = 1536  # (mu+sd) pooled dim you used

def _norm(s): return str(s).replace("\\", "/")

def load_csv_safe(p: Path, needed=None):
    if not p.exists(): return None
    try:
        df = pd.read_csv(p)
        if needed:
            for c in needed:
                if c not in df.columns: df[c] = ""
        return df
    except Exception as e:
        print(f"[warn] failed to read {p}: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r'Project root, e.g. "G:\My Drive\hindi_dfake"')
    args = ap.parse_args()

    ROOT = Path(args.root)
    META = ROOT / "metadata"
    PROC = ROOT / "processed"
    WAV  = PROC / "wav" / PROFILE
    FEAT_ROOT = PROC / "features" / "ptm"

    print(f"[paths] ROOT={ROOT}")
    print(f"        WAV ={WAV} (exists={WAV.exists()})")
    print(f"        FEAT={FEAT_ROOT} (exists={FEAT_ROOT.exists()})")
    print(f"        META={META} (exists={META.exists()})")

    # runners' subset files
    sub_v2   = META / "ptm_subset.v2.csv"         # wav2vec2 runner
    sub_v2f  = META / "ptm_subset.v2.fixed.csv"   # hubert runner
    manifest = META / "features_manifest.csv"

    df_v2  = load_csv_safe(sub_v2,  needed=["path_audio","profile"])
    df_v2f = load_csv_safe(sub_v2f, needed=["path_audio","profile"])
    df_mani = load_csv_safe(manifest, needed=["path_audio","profile","ptm","vec_path","dim","seconds"])

    # expected (train/val pool) = union of subset files, profile=strong only
    frames = []
    if df_v2 is not None:
        frames.append(df_v2[df_v2["profile"].astype(str).str.lower()==PROFILE])
    if df_v2f is not None:
        frames.append(df_v2f[df_v2f["profile"].astype(str).str.lower()==PROFILE])
    if frames:
        df_expected = pd.concat(frames, ignore_index=True)
        # normalize path strings; take unique
        df_expected["path_audio"] = df_expected["path_audio"].astype(str)
        df_expected["path_audio_n"] = df_expected["path_audio"].map(lambda s: _norm(s))
        df_expected = df_expected.drop_duplicates(subset=["path_audio_n"])
    else:
        df_expected = pd.DataFrame(columns=["path_audio","path_audio_n","profile"])
    print(f"\n[expected train/val pool] unique audios (profile={PROFILE}) = {len(df_expected):,}")

    # helper to resolve expected vec path (runners mirror processed/wav/<profile>/tail)
    def vec_path_for(ptm_name: str, path_audio: str) -> Path:
        pa = Path(path_audio)
        try:
            tail = pa.resolve().relative_to(WAV.resolve())
        except Exception:
            # fallback to filename only (rare)
            tail = Path(pa.name)
        return FEAT_ROOT / ptm_name / tail.with_suffix(".npy")

    # audit per PTM for expected pool
    for ptm in PTMS:
        print(f"\n===== [{ptm} :: expected pool coverage] =====")
        if df_expected.empty:
            print("  (no subset CSVs found — skipping)")
            continue

        exp_vecs = df_expected["path_audio"].astype(str).map(lambda s: str(vec_path_for(ptm, s)))
        exp_vecs = pd.Series(exp_vecs.unique())  # unique per PTM
        print(f"  expected vectors: {len(exp_vecs):,}")

        exist_mask = exp_vecs.map(lambda p: Path(p).exists())
        n_exist = int(exist_mask.sum())
        print(f"  on-disk present : {n_exist:,}")
        missing = exp_vecs[~exist_mask].tolist()

        # check readability / dtype / shape
        unreadable = []
        bad_dtype = []
        bad_shape = []
        for p in exp_vecs[exist_mask]:
            try:
                arr = np.load(p, mmap_mode="r")
                a = np.array(arr)
                if a.dtype != np.float16 and a.dtype != np.float32:
                    bad_dtype.append(f"{p} ({a.dtype})")
                    continue
                # allow (D,) or (T,D) with D==EXPECT_DIM
                if a.ndim == 1 and a.shape[0] == EXPECT_DIM:
                    pass
                elif a.ndim == 2 and a.shape[-1] == EXPECT_DIM:
                    pass
                else:
                    bad_shape.append(f"{p} {tuple(a.shape)}")
            except Exception as e:
                unreadable.append(f"{p} ({str(e).splitlines()[0]})")

        print(f"  unreadable       : {len(unreadable):,}")
        print(f"  bad dtype        : {len(bad_dtype):,}")
        print(f"  bad shape        : {len(bad_shape):,}")

        # manifest cross-check
        in_manifest = 0
        if df_mani is not None:
            mani_set = set(df_mani.loc[df_mani["ptm"]==ptm, "vec_path"].astype(str))
            in_manifest = int(exp_vecs.map(lambda s: s in mani_set).sum())
        print(f"  in features_manifest.csv : {in_manifest:,}")

        # samples
        def show(title, lst):
            if not lst: return
            print(f"  examples {title}:")
            for x in lst[:5]:
                print("    ·", x)
        show("MISSING", missing)
        show("UNREADABLE", unreadable)
        show("BAD_DTYPE", bad_dtype)
        show("BAD_SHAPE", bad_shape)

    # ---------- test CSVs sanity (the ones MoE uses at testing time) ----------
    tests_dir = META / "tests"
    test_csvs = []
    for name in ["test_mms.strong.ptm2.csv", "test_edge.strong.ptm2.csv", "split_test.ptm2.csv"]:
        p = (tests_dir / name) if "test_" in name else (META / name)
        if p.exists(): test_csvs.append(p)

    if test_csvs:
        print("\n===== [test ptm2.csv integrity] =====")
    for tcsv in test_csvs:
        df = load_csv_safe(tcsv)
        if df is None:
            continue
        # detect vec columns
        vec_cols = [c for c in df.columns if df[c].dtype == object and (df[c].astype(str).str.endswith(".npy").mean() > 0.6)]
        print(f"\n  [{tcsv.name}] rows={len(df)} | vec_cols={vec_cols}")
        for col in vec_cols:
            paths = df[col].astype(str).tolist()
            exists = sum(1 for p in paths if Path(p).exists())
            print(f"    - {col}: exists {exists:,} / {len(paths):,}")
            # spot-check a few problems
            miss = [p for p in paths if not Path(p).exists()]
            if miss[:3]:
                print("      examples missing:")
                for m in miss[:3]: print("        ·", m)

    print("\n[done] read-only audit complete.")

if __name__ == "__main__":
    main()
