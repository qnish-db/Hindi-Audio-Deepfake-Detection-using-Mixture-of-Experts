# check_env.py  — env checks + inference + optional MMS calibration
from pathlib import Path
import os, json, argparse, random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------- ROOT & fixed paths ----------
ROOT = Path(r"G:\My Drive\hindi_dfake")
META = ROOT / "metadata"

TEST_CSV_MAP = {
    "core": META / "split_test.ptm2.csv",
    "edge": META / "tests" / "test_edge.strong.ptm2.csv",
    "mms" : META / "tests" / "test_mms.strong.ptm2.csv",
}

# ---------- small utils ----------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _norm(s: str) -> str:
    return str(s).replace("\\", "/")

def resolve_test_csv(args, ckpt_cfg):
    if args.test_csv:
        return Path(args.test_csv)
    key = (args.test_set or "").lower().strip()
    if key and key in TEST_CSV_MAP and TEST_CSV_MAP[key].exists():
        return TEST_CSV_MAP[key]
    # fallback to whatever was used during training
    p = ckpt_cfg.get("test_csv", "")
    return Path(p) if p else TEST_CSV_MAP.get("core", META / "split_test.ptm2.csv")

def resolve_val_csv(ckpt_cfg):
    p = ckpt_cfg.get("val_csv", str(META / "split_val.ptm2.csv"))
    return Path(p)

def resolve_ptm_columns(csv_path: Path, ptm_list: List[str]) -> Dict[str, str]:
    dfh = pd.read_csv(csv_path, nrows=50)
    cols = list(dfh.columns)
    def norm(s): return s.lower().replace("-", "").replace("_", "")
    npy_cols = []
    for c in cols:
        if dfh[c].dtype == object:
            vals = dfh[c].dropna().astype(str)
            if not vals.empty and (vals.str.endswith(".npy").mean() > 0.8):
                npy_cols.append(c)
    col_map = {}
    for ptm in ptm_list:
        target = norm(ptm)
        cands = [c for c in npy_cols if target in norm(c)]
        if len(cands) == 1: col_map[ptm] = cands[0]
        elif len(cands) > 1:
            cands.sort(key=len); col_map[ptm] = cands[0]
        else:
            # fallback to any unused .npy-like col
            fallback = [c for c in npy_cols if c not in col_map.values()]
            if not fallback:
                raise ValueError(f"Can't find .npy column for PTM '{ptm}' in {csv_path}. "
                                 f"Available: {cols}")
            col_map[ptm] = fallback[0]
    print("[ptm columns]", " | ".join([f"{k} -> {v}" for k, v in col_map.items()]))
    return col_map

def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    # returns (eer, thr_at_eer)
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return 0.5, 0.0
    tp = fp = 0; fn = P; tn = N
    best_diff = 1.0; best_eer = 1.0; thr = s[0] if len(s) else 0.0
    prev = np.inf
    for i in range(len(s)):
        si, yi = s[i], y[i]
        if si != prev:
            fpr = fp / N; fnr = fn / P
            diff = abs(fpr - fnr)
            if diff < best_diff:
                best_diff = diff
                best_eer = (fpr + fnr) / 2.0
                thr = prev
            prev = si
        if yi == 1:
            tp += 1; fn -= 1
        else:
            fp += 1; tn -= 1
    fpr = fp / N; fnr = fn / P
    diff = abs(fpr - fnr)
    if diff < best_diff:
        best_eer = (fpr + fnr) / 2.0
        thr = s[-1]
    return float(best_eer), float(thr)

def acc_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    pred = (scores >= thr).astype(np.int32)
    return float((pred == labels).mean())

def confusion(scores: np.ndarray, labels: np.ndarray, thr: float):
    pred = (scores >= thr).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn

# ---------- MoE model (same as training) ----------
class SmallExpert(nn.Module):
    def __init__(self, in_dim=1536, hidden=512, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 2)
        )
    def forward(self, x): return self.net(x)

class TinyGate(nn.Module):
    def __init__(self, in_dim_concat, hidden=64, drop=0.1, n_experts=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim_concat),
            nn.Linear(in_dim_concat, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, n_experts)
        )
    def forward(self, x_concat):
        w = self.net(x_concat)
        return torch.softmax(w, dim=1)

class MoEModel(nn.Module):
    def __init__(self, ptms: List[str], in_dim_each=1536, expert_hidden=512, expert_drop=0.2,
                 gate_hidden=64, gate_drop=0.1):
        super().__init__()
        self.ptms = ptms
        self.in_dim_each = in_dim_each
        self.experts = nn.ModuleDict({ptm: SmallExpert(in_dim_each, expert_hidden, expert_drop)
                                      for ptm in ptms})
        self.gate = TinyGate(in_dim_each * len(ptms), gate_hidden, gate_drop, len(ptms))
    def forward(self, x_dict: Dict[str, torch.Tensor]):
        xs = [x_dict[ptm] for ptm in self.ptms]
        x_concat = torch.cat(xs, dim=1)
        gate_w = self.gate(x_concat)                  # (B,E)
        expert_logits = torch.stack([self.experts[p](x) for p, x in zip(self.ptms, xs)], dim=1)  # (B,E,2)
        final_logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)  # (B,2)
        return final_logits

# ---------- dataset that reads .npy paths ----------
class PTMVecSet(Dataset):
    def __init__(self, csv_path: Path, ptms: List[str], ptm_cols: Dict[str, str]):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.ptms = ptms
        self.cols = ptm_cols
        if "label" not in self.df.columns:
            raise ValueError(f"Missing 'label' in {csv_path}")
        for p in ptms:
            if self.cols[p] not in self.df.columns:
                raise ValueError(f"Missing column {self.cols[p]} for PTM {p}")
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x = {}
        for p in self.ptms:
            v = np.load(r[self.cols[p]])
            if v.dtype != np.float32: v = v.astype(np.float32, copy=False)
            x[p] = torch.from_numpy(v)
        y = int(r["label"])
        return x, torch.tensor(y, dtype=torch.long)

def scores_on_loader(model, loader, device):
    model.eval()
    outs, labs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xdev = {k: v.to(device) for k, v in xb.items()}
            logits = model(xdev)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            outs.append(probs.cpu().numpy())
            labs.append(yb.numpy().astype(np.int32))
    return np.concatenate(outs), np.concatenate(labs)

def build_loader(csv_path, ptms, ptm_cols, batch=512, workers=0):
    ds = PTMVecSet(csv_path, ptms, ptm_cols)
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True,
                    collate_fn=lambda bl: (
                        {p: torch.stack([b[0][p] for b in bl], dim=0) for p in ptms},
                        torch.stack([b[1] for b in bl], dim=0),
                    ))
    return ds, ld

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="Path to a saved .pt checkpoint for inference")
    ap.add_argument("--test-set", type=str, default="mms", help="mms|edge|core|custom")
    ap.add_argument("--test-csv", type=str, default="", help="Path to a specific test csv (overrides --test-set)")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (vectors are tiny; 0 is fine)")
    ap.add_argument("--calibrate", action="store_true", help="Split MMS into dev/holdout and tune threshold on dev")
    ap.add_argument("--calib-frac", type=float, default=0.2, help="Dev fraction for calibration (e.g., 0.2)")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Quick env print
    print(f"[paths] ROOT={ROOT}")

    # If only doing the old checks (no ckpt), keep it minimal
    if not args.ckpt:
        # Minimal legacy checks that you had (counts/shapes). Keep short:
        # Scan MMS set presence & feature columns
        test_csv = TEST_CSV_MAP.get("mms", None)
        if test_csv and test_csv.exists():
            df = pd.read_csv(test_csv)
            print(f"[mms] rows={len(df)} | real={(df['label']==0).sum()} | fake={(df['label']==1).sum()} | "
                  f"speakers={df['speaker_id'].astype(str).nunique()}")
            # Simple sanity on shapes for first 1000
            ptm_cols = resolve_ptm_columns(test_csv, ["wav2vec2-base","hubert-base"])
            ok = []
            for c in ptm_cols.values():
                sub = df[c].dropna().astype(str).head(1000)
                shapes = []
                miss = bad = 0
                for p in sub:
                    if not os.path.exists(p): miss += 1; continue
                    try:
                        v = np.load(p, mmap_mode="r")
                        shapes.append(tuple(v.shape))
                        if np.isnan(v).any() or np.isinf(v).any(): bad += 1
                    except:
                        bad += 1
                print(f"[{c}] missing={miss} bad={bad} shapes_sample={dict(pd.Series(shapes).value_counts().head(3))}")
                ok.append(miss==0 and bad==0)
            if all(ok): print("[status] MMS feature set looks GOOD ✅")
        else:
            print("[warn] MMS test csv not found; pass --ckpt to run inference.")
        return

    # ---------- load checkpoint ----------
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    ptms = cfg.get("ptms", ["wav2vec2-base","hubert-base"])
    in_dim = int(cfg.get("ptm_dim", 1536))
    expert_hidden = int(cfg.get("expert_hidden", 512))
    expert_drop   = float(cfg.get("expert_dropout", 0.2))
    gate_hidden   = int(cfg.get("gate_hidden", 64))
    gate_drop     = float(cfg.get("gate_dropout", 0.1))

    # data paths
    test_csv = resolve_test_csv(args, cfg)
    val_csv  = resolve_val_csv(cfg)

    # resolve columns
    ptm_cols_test = resolve_ptm_columns(test_csv, ptms)
    ptm_cols_val  = resolve_ptm_columns(val_csv,  ptms)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} | cuda_available={torch.cuda.is_available()} | "
          f"gpu={(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')}")

    # model
    model = MoEModel(ptms, in_dim, expert_hidden, expert_drop, gate_hidden, gate_drop).to(device)
    model.load_state_dict(ckpt["model"])

    # loaders
    _, val_loader  = build_loader(val_csv,  ptms, ptm_cols_val,  batch=512, workers=args.num_workers)
    _, test_loader = build_loader(test_csv, ptms, ptm_cols_test, batch=512, workers=args.num_workers)

    # scores
    val_scores, val_labels   = scores_on_loader(model, val_loader,  device)
    test_scores, test_labels = scores_on_loader(model, test_loader, device)

    # thresholds: 0.5 and Val-EER
    val_eer, val_thr = eer_from_scores(val_scores, val_labels)
    acc05     = acc_at(test_scores, test_labels, 0.5)
    acc_valth = acc_at(test_scores, test_labels, val_thr)

    print("\n[VAL]")
    print(f"  EER={val_eer:.4f} | thr@eer={val_thr:.6f}")

    print("\n[MMS TEST] (no calibration)")
    print(f"  Acc@0.5     = {acc05:.4f}")
    print(f"  Acc@val_thr = {acc_valth:.4f}  (val_thr={val_thr:.6f})")
    tp, fp, tn, fn = confusion(test_scores, test_labels, val_thr)
    print("  Confusion @ val_thr:")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # optional calibration on MMS dev
    if args.calibrate:
        df = pd.read_csv(test_csv).reset_index(drop=True)
        # stratified split by label
        frac = min(max(args.calib_frac, 0.05), 0.5)
        idx_pos = df.index[df["label"]==1].tolist()
        idx_neg = df.index[df["label"]==0].tolist()
        n_pos = max(1, int(len(idx_pos)*frac))
        n_neg = max(1, int(len(idx_neg)*frac))
        rng = np.random.default_rng(args.seed)
        dev_idx = set(rng.choice(idx_pos, n_pos, replace=False).tolist() +
                      rng.choice(idx_neg, n_neg, replace=False).tolist())
        hold_idx = [i for i in range(len(df)) if i not in dev_idx]

        # build tensors once from already computed scores
        dev_scores   = test_scores[list(dev_idx)]
        dev_labels   = test_labels[list(dev_idx)]
        hold_scores  = test_scores[hold_idx]
        hold_labels  = test_labels[hold_idx]

        # find best acc threshold on dev
        # use unique scores + small grid around
        uniq = np.unique(dev_scores)
        cand = np.linspace(0.0, 1.0, 401)
        thr_grid = np.unique(np.concatenate([uniq, cand]))
        best_acc = -1.0; best_thr = 0.5
        for t in thr_grid:
            a = acc_at(dev_scores, dev_labels, t)
            if a > best_acc:
                best_acc, best_thr = a, t

        # evaluate on holdout
        acc_hold = acc_at(hold_scores, hold_labels, best_thr)
        tp2, fp2, tn2, fn2 = confusion(hold_scores, hold_labels, best_thr)
        print("\n[MMS calibration]")
        print(f"  dev_frac={frac:.2f} | dev_n={len(dev_scores)} | holdout_n={len(hold_scores)}")
        print(f"  Calibrated thr={best_thr:.6f} (dev best acc={best_acc:.4f})")
        print(f"  Holdout Acc   = {acc_hold:.4f}")
        print("  Confusion @ calibrated thr (holdout):")
        print(f"    TP={tp2}  FP={fp2}  TN={tn2}  FN={fn2}")

        # save small note (optional)
        outj = META / "runs" / "mms_calibration.json"
        outj.parent.mkdir(parents=True, exist_ok=True)
        with open(outj, "w", encoding="utf-8") as f:
            json.dump({
                "ckpt": str(args.ckpt),
                "test_csv": str(test_csv),
                "val_thr": float(val_thr),
                "calib_frac": float(frac),
                "calibrated_thr": float(best_thr),
                "dev_size": int(len(dev_scores)),
                "holdout_size": int(len(hold_scores)),
                "acc_holdout": float(acc_hold),
            }, f, indent=2)
        print(f"[saved] {outj}")

if __name__ == "__main__":
    main()
