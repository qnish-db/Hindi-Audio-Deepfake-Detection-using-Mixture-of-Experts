# audit_fs_universe_v2.py
# FAST filesystem audit over processed audio & PTM features.
# No split CSVs. Fixes tail normalization so masters ↔ processed/features match.

from __future__ import annotations
from pathlib import Path
import argparse, os, hashlib
from collections import Counter, defaultdict
import pandas as pd

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Filesystem-only audit (processed wav / PTM features).")
    ap.add_argument("--root", required=True, help=r'Root, e.g. "G:\My Drive\hindi_dfake"')
    ap.add_argument("--profile", default="strong", help="processed/wav/<profile> (default: strong)")
    ap.add_argument("--ptms", nargs="+", default=["wav2vec2-base","hubert-base"],
                    help="PTM dirs under processed/features/ptm to require for intersection stats")
    ap.add_argument("--real_csv", default="metadata/master_real.csv")
    ap.add_argument("--fake_csv", default="metadata/master_fake.csv")
    ap.add_argument("--fleurs_csv", default="metadata/thirdparty_real_test.fleurs.csv")
    ap.add_argument("--test-sources", nargs="*", default=[],
                    help='Optional list of source names (e.g. CommonVoice IndicVoices_R) to treat as "test-ish"')
    ap.add_argument("--test-fake-types", nargs="*", default=[],
                    help='Optional fake_type filters for test (e.g. tts_mms)')
    ap.add_argument("--max_print", type=int, default=12)
    ap.add_argument("--list", action="store_true", help="print sample tails for gaps")
    return ap.parse_args()

# ---------------- utils ----------------
def _norm(s:str)->str: return str(s).replace("\\","/")

def tail_after_raw(path_str:str)->str:
    """Return tail *after* '/raw/' if present; otherwise drop any leading 'processed/wav/<profile>/'
       and also strip a leading 'raw/' if it remains. Final form: e.g. 'real_clean/ivr/foo.wav'."""
    s = _norm(path_str)
    if "/raw/" in s:
        t = s.split("/raw/",1)[1]
    else:
        # maybe it's already under processed/wav/<profile>[/raw]/...
        parts = s.split("/processed/wav/",1)
        if len(parts)==2:
            t = parts[1].split("/",1)[1] if "/" in parts[1] else parts[1]  # drop '<profile>/'
        else:
            t = os.path.basename(s)
    # normalize: strip leading 'raw/' if present
    if t.startswith("raw/"): t = t[4:]
    return t

def scan_audio_tails(root:Path, profile:str)->set[str]:
    base = (root/"processed"/"wav"/profile).resolve()
    tails = set()
    if not base.exists(): return tails
    for dp, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".wav"):
                full = (Path(dp)/f).resolve()
                tails.add(tail_after_raw(str(full)))
    return tails

def scan_features_tails(root:Path, ptm:str)->set[str]:
    base = (root/"processed"/"features"/"ptm"/ptm).resolve()
    tails = set()
    if not base.exists(): return tails
    for dp, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".npy"):
                full = (Path(dp)/f).resolve()
                t = _norm(str(full))
                # replace PTM base & .npy with wav-like tail; then normalize
                t = t.split(f"/ptm/{ptm}/",1)[1]
                if t.lower().endswith(".npy"): t = t[:-4] + ".wav"
                tails.add(tail_after_raw(t))
    return tails

def stable_pick(seq, k, seed=2025):
    if len(seq)<=k: return list(seq)
    def H(x): return int(hashlib.sha1((str(x)+str(seed)).encode()).hexdigest()[:8],16)
    return [x for x,_ in sorted(((x,H(x)) for x in seq))[:k]]

# ---------------- masters ----------------
def load_masters(real_csv:Path, fake_csv:Path):
    if not real_csv.exists() or not fake_csv.exists():
        raise FileNotFoundError("master_real.csv or master_fake.csv missing.")
    real = pd.read_csv(real_csv)
    fake = pd.read_csv(fake_csv)

    for df in (real, fake):
        if "path" not in df.columns:
            raise KeyError("master CSV missing 'path'")
        df["path"] = df["path"].astype(str)
        df["tail"] = df["path"].map(tail_after_raw)

    # ensure columns
    for c in ["utt_id","speaker_id","source"]:
        if c not in real.columns: real[c] = ""
    for c in ["utt_id","speaker_id","source","voice","tts_model","fake_type"]:
        if c not in fake.columns: fake[c] = ""

    real["label"] = 0
    fake["label"] = 1

    r_idx = real.drop_duplicates("tail").set_index("tail")
    f_idx = fake.drop_duplicates("tail").set_index("tail")
    return r_idx, f_idx

def label_meta_for_tail(t:str, r_idx, f_idx):
    if t in r_idx.index:
        row = r_idx.loc[t]
        return 0, str(row.get("speaker_id","")) or "unknown_0", str(row.get("source","")), ""
    if t in f_idx.index:
        row = f_idx.loc[t]
        voice = str(row.get("voice","")) or "voice?"
        tts   = str(row.get("tts_model","")) or "tts?"
        ftype = str(row.get("fake_type","")).lower()
        return 1, f"{voice}|{tts}", str(row.get("source","")), ftype
    return None, "", "", ""

# ---------------- reporting ----------------
def summarize(name:str, tails:set[str], r_idx, f_idx):
    n = len(tails)
    r=f=mms=0
    spk_real, spk_fake = Counter(), Counter()
    src_real, src_fake = Counter(), Counter()
    for t in tails:
        lab, spk, src, ftype = label_meta_for_tail(t, r_idx, f_idx)
        if lab is None: continue
        if lab==0:
            r += 1; spk_real[spk]+=1; src_real[src]+=1
        else:
            f += 1; spk_fake[spk]+=1; src_fake[src]+=1
            if ftype=="tts_mms": mms += 1

    def mu_med(counter:Counter):
        import numpy as np
        if not counter: return (0.0,0.0)
        arr = list(counter.values())
        return float(np.mean(arr)), float(np.median(arr))

    r_mu,r_med = mu_med(spk_real); f_mu,f_med = mu_med(spk_fake)

    print(f"\n=== {name} ===")
    print(f"total files: {n:,} | labeled: {r+f:,} (real={r:,}, fake={f:,} | fake tts_mms={mms:,})")
    print(f"speakers: real={len(spk_real):,} (μ/med={r_mu:.1f}/{r_med:.1f}), "
          f"fake={len(spk_fake):,} (μ/med={f_mu:.1f}/{f_med:.1f})")
    if r or f:
        top_r = ', '.join(f"{k}:{v}" for k,v in src_real.most_common(8)) or "(none)"
        top_f = ', '.join(f"{k}:{v}" for k,v in src_fake.most_common(8)) or "(none)"
        print(f"sources (real): {top_r}")
        print(f"sources (fake): {top_f}")
    return {"n":n, "real":r, "fake":f, "mms":mms,
            "spk_real":spk_real, "spk_fake":spk_fake,
            "src_real":src_real, "src_fake":src_fake}

def project_test_and_train(audio_tails:set[str], r_idx, f_idx,
                           test_sources:list[str], test_fake_types:list[str]):
    """Estimate a 'test-ish' slice from the *audio universe* by source/fake_type filters.
       Returns (test_set, trainval_set) as sets of tails (only labeled ones)."""
    test_set = set()
    trainval_set = set()
    for t in audio_tails:
        lab, spk, src, ftype = label_meta_for_tail(t, r_idx, f_idx)
        if lab is None:
            continue
        in_test = False
        if lab==0:
            # real goes to test if its source in test_sources (e.g., FLEURS handled separately below)
            if src in set(test_sources):
                in_test = True
        else:
            if (not test_fake_types) or (ftype in set(test_fake_types)):
                if (not test_sources) or (src in set(test_sources)):
                    in_test = True
        (test_set if in_test else trainval_set).add(t)
    return test_set, trainval_set

# ---------------- main ----------------
def main():
    args = parse_args()
    ROOT = Path(args.root)

    r_idx, f_idx = load_masters(ROOT/args.real_csv, ROOT/args.fake_csv)

    # ---- scan universes
    audio_tails = scan_audio_tails(ROOT, args.profile)
    print(f"[scan] audio @ processed/wav/{args.profile}: {len(audio_tails):,} wav")

    ptm_sets = {}
    for ptm in args.ptms:
        s = scan_features_tails(ROOT, ptm)
        ptm_sets[ptm] = s
        print(f"[scan] features {ptm}: {len(s):,} npy→wav tails")
    inter = set.intersection(*ptm_sets.values()) if ptm_sets else set()
    print(f"[scan] features intersection across {args.ptms}: {len(inter):,}")

    audio_only = audio_tails - inter
    inter_only = inter - audio_tails
    print(f"[coverage] audio with ALL PTMs: {len(audio_tails & inter):,}/{len(audio_tails):,}")
    print(f"[coverage] features-only (no audio): {len(inter_only):,}")
    print(f"[coverage] audio-only (missing some PTM): {len(audio_only):,}")
    if args.list:
        if inter_only:
            print("  eg features-only:", *stable_pick(list(inter_only), min(5,len(inter_only))), sep="\n   - ")
        if audio_only:
            print("  eg audio-only   :", *stable_pick(list(audio_only), min(5,len(audio_only))), sep="\n   - ")

    # ---- label summaries
    stats_audio = summarize(f"PREPROCESSED AUDIO (processed/wav/{args.profile})",
                            audio_tails, r_idx, f_idx)
    stats_feat  = summarize(f"FEATURES (ALL PTMs: {', '.join(args.ptms)})",
                            inter, r_idx, f_idx)

    # ---- FLEURS cross-check (REAL test pool)
    fleurs_p = ROOT/args.fleurs_csv
    fleurs_set = set()
    if fleurs_p.exists():
        dfF = pd.read_csv(fleurs_p)
        if "path" in dfF.columns:
            fleurs_set = set(dfF["path"].astype(str).map(tail_after_raw))
            in_audio = len(fleurs_set & audio_tails)
            in_feat  = len(fleurs_set & inter)
            print(f"\n[FLEURS] rows={len(fleurs_set):,} | present in audio={in_audio:,} | in ALL-PTM features={in_feat:,}")
        else:
            print("\n[FLEURS] CSV missing 'path' — skipped.")
    else:
        print("\n[FLEURS] CSV not found — skipped.")

    # ---- Optional projection: “test-ish” by source/fake_type (e.g., CommonVoice + IndicVoices_R + tts_mms)
    test_set, trainval_set = set(), set()
    if args.test_sources or args.test_fake_types:
        test_set, trainval_set = project_test_and_train(audio_tails, r_idx, f_idx,
                                                        args.test_sources, args.test_fake_types)
        # force-add FLEURS real into test bucket if present
        test_set |= (fleurs_set & audio_tails)
        # keep only labeled items in the union
        labeled_audio = {t for t in audio_tails if label_meta_for_tail(t, r_idx, f_idx)[0] is not None}
        trainval_set = (labeled_audio - test_set)

        # count label breakdowns
        def count_set(S:set[str]):
            rr=ff=mm=0; srcs=Counter()
            for t in S:
                lab, _, src, ftype = label_meta_for_tail(t, r_idx, f_idx)
                if lab==0: rr+=1; srcs[src]+=1
                elif lab==1: ff+=1; srcs[src]+=1
                if lab==1 and ftype=="tts_mms": mm+=1
            return rr,ff,mm,srcs

        rT,fT,mT,sT = count_set(test_set)
        rTR,fTR,mTR,sTR = count_set(trainval_set)

        print("\n--- PROJECTED BUCKETS (from *audio* universe) ---")
        print(f"Test-ish sources={args.test_sources or 'ANY'} | fake_types={args.test_fake_types or 'ANY'} + FLEURS(real)")
        print(f"  TEST total={len(test_set):,} | real={rT:,} | fake={fT:,} (tts_mms={mT:,})")
        print(f"  TRAIN+VAL total={len(trainval_set):,} | real={rTR:,} | fake={fTR:,} (tts_mms={mTR:,})")
        print("  top test sources :", ', '.join(f'{k}:{v}' for k,v in sT.most_common(8)) or '(none)')
        print("  top train/val src:", ', '.join(f'{k}:{v}' for k,v in sTR.most_common(8)) or '(none)')

    print("\n[ok] filesystem audit complete.")

if __name__ == "__main__":
    main()
