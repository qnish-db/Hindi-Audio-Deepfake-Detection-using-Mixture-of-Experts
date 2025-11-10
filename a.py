# a.py — quick MMS test evaluator (auto-finds checkpoint)
# - Auto-discovers ROOT (defaults to G:\My Drive\hindi_dfake)
# - Auto-discovers best checkpoint in ROOT/checkpoints or ./checkpoints
# - Loads MoE model, runs inference on MMS test CSV
# - Prints Acc@0.5, EER + thr@eer, and confusion matrices

from pathlib import Path
import sys, re, json, math, argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Defaults / CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=r"G:\My Drive\hindi_dfake",
                    help="Project ROOT (Drive).")
    ap.add_argument("--mms-csv", type=str, default=None,
                    help="Path to MMS test CSV (defaults to <ROOT>/metadata/tests/test_mms.strong.ptm2.csv).")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to best checkpoint; if omitted, auto-discover.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--ptms", type=str, default="wav2vec2-base,hubert-base")
    ap.add_argument("--ptm-dim", type=int, default=1536)
    return ap.parse_args()

# =========================
# Path helpers
# =========================
def find_ckpt(ROOT: Path, preferred_name="moe_ptm2_simpleNN_v1_best.pt") -> Path | None:
    # First try exact expected locations
    candidates = [
        ROOT / "checkpoints" / preferred_name,
        Path.cwd() / "checkpoints" / preferred_name,
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: newest "*best.pt" under common dirs
    search_dirs = [ROOT / "checkpoints", Path.cwd() / "checkpoints", ROOT, Path.cwd()]
    found = []
    pat = re.compile(r".*best\.pt$", re.IGNORECASE)
    for d in search_dirs:
        if d.exists():
            for p in d.rglob("*.pt"):
                if pat.match(str(p)):
                    found.append(p)
    if not found:
        return None
    # Prefer exact name if present; else pick most recently modified
    exact = [p for p in found if p.name == preferred_name]
    if exact:
        return sorted(exact, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[0]

# =========================
# Column resolver (same logic as train)
# =========================
def resolve_ptm_columns(csv_path: str, ptm_list: List[str]) -> Dict[str, str]:
    df_head = pd.read_csv(csv_path, nrows=50)
    cols = list(df_head.columns)
    col_map: Dict[str, str] = {}

    def norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")

    npy_like_cols = []
    for c in cols:
        if df_head[c].dtype == object:
            vals = df_head[c].dropna().astype(str)
            if not vals.empty and (vals.str.endswith(".npy").mean() > 0.8):
                npy_like_cols.append(c)

    for ptm in ptm_list:
        target = norm(ptm)
        candidates = [c for c in npy_like_cols if target in norm(c)]
        if len(candidates) == 1:
            col_map[ptm] = candidates[0]
        elif len(candidates) > 1:
            candidates.sort(key=len)
            col_map[ptm] = candidates[0]
        else:
            fallback = [c for c in npy_like_cols if c not in col_map.values()]
            if fallback:
                col_map[ptm] = fallback[0]
            else:
                raise ValueError(
                    f"Could not find a column with .npy paths for PTM '{ptm}' in {csv_path}.\n"
                    f"Available columns: {cols}"
                )
    print("[ptm columns]", " | ".join([f"{k} -> {v}" for k, v in col_map.items()]))
    return col_map

# =========================
# Dataset / Loader
# =========================
def load_vec(path: str) -> np.ndarray:
    v = np.load(path)
    if v.dtype != np.float32:
        v = v.astype(np.float32, copy=False)
    return v

class PTMDataset(Dataset):
    def __init__(self, csv_path: str, ptm_list: List[str], ptm_columns: Dict[str, str]):
        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError(f"Missing 'label' column in {csv_path}")
        # thin copy
        self.df = df.reset_index(drop=True)
        self.ptms = ptm_list
        self.ptm_cols = ptm_columns
        for ptm in self.ptms:
            col = self.ptm_cols[ptm]
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' for PTM '{ptm}' in {csv_path}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = {}
        for ptm in self.ptms:
            p = row[self.ptm_cols[ptm]]
            xs[ptm] = torch.from_numpy(load_vec(p))
        y = int(row["label"])
        return {"x": xs, "y": torch.tensor(y, dtype=torch.long)}

def collate_fn(batch_list):
    ptm_names = list(batch_list[0]["x"].keys())
    xs = {ptm: torch.stack([b["x"][ptm] for b in batch_list], dim=0) for ptm in ptm_names}
    y = torch.stack([b["y"] for b in batch_list], dim=0)
    return {"x": xs, "y": y}

# =========================
# Model (same as train)
# =========================
class SmallExpert(nn.Module):
    def __init__(self, in_dim=1536, hidden=512, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 2),
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
            nn.Linear(hidden, n_experts),
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
        self.experts = nn.ModuleDict({
            ptm: SmallExpert(in_dim_each, expert_hidden, expert_drop) for ptm in ptms
        })
        self.gate = TinyGate(
            in_dim_concat=in_dim_each * len(ptms),
            hidden=gate_hidden, drop=gate_drop, n_experts=len(ptms)
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]):
        xs = [x_dict[p] for p in self.ptms]
        x_concat = torch.cat(xs, dim=1)
        gate_w = self.gate(x_concat)                 # (B,E)
        expert_logits = torch.stack([self.experts[p](x_dict[p]) for p in self.ptms], dim=1)  # (B,E,2)
        logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)  # (B,2)
        return logits, expert_logits, gate_w

# =========================
# Metrics
# =========================
@torch.no_grad()
def compute_scores_and_labels(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y  = batch["y"].to(device, non_blocking=True)
        logits, _, _ = model(xs)
        prob_fake = torch.softmax(logits, dim=1)[:, 1]
        scores.append(prob_fake.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels).astype(np.int32)

def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-scores)
    scores = scores[order]; labels = labels[order]
    P = (labels == 1).sum(); N = (labels == 0).sum()
    if P == 0 or N == 0:
        return 0.5, 0.0
    tp = fp = 0; fn = P; tn = N
    best_diff = 1.0; eer = 1.0
    thr_at_eer = scores[0] if len(scores) else 0.0
    prev_s = float("inf")
    for i in range(len(scores)):
        s, y = scores[i], labels[i]
        if s != prev_s:
            fpr = fp / N; fnr = fn / P
            diff = abs(fpr - fnr)
            if diff < best_diff:
                best_diff = diff
                eer = (fpr + fnr) / 2.0
                thr_at_eer = prev_s
            prev_s = s
        if y == 1:
            tp += 1; fn -= 1
        else:
            fp += 1; tn -= 1
    fpr = fp / N; fnr = fn / P
    diff = abs(fpr - fnr)
    if diff < best_diff:
        eer = (fpr + fnr) / 2.0
        thr_at_eer = scores[-1]
    return float(eer), float(thr_at_eer)

def confusion(scores: np.ndarray, labels: np.ndarray, thr: float):
    pred = (scores >= thr).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(1, len(labels))
    return tp, fp, tn, fn, acc

def best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    # grid search 200 bins between [min, max]
    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        return hi, ((labels == (scores >= hi)).mean())
    thrs = np.linspace(lo, hi, 200, dtype=np.float32)
    best_acc, best_thr = 0.0, thrs[0]
    for t in thrs:
        _, _, _, _, acc = confusion(scores, labels, float(t))
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)
    return best_thr, best_acc

# =========================
# Main
# =========================
def main():
    args = parse_args()
    ROOT = Path(args.root)
    MMS_CSV = Path(args.mms_csv) if args.mms_csv else (ROOT / "metadata" / "tests" / "test_mms.strong.ptm2.csv")
    CKPT = Path(args.ckpt) if args.ckpt else find_ckpt(ROOT)

    print("[paths]")
    print("  ROOT   :", ROOT)
    print("  MMS_CSV:", MMS_CSV, "exists=", MMS_CSV.exists())
    print("  CKPT   :", CKPT if CKPT else "(not found)")

    if not MMS_CSV.exists():
        raise FileNotFoundError(f"MMS CSV not found: {MMS_CSV}")
    if CKPT is None or not CKPT.exists():
        raise FileNotFoundError("Best checkpoint not found. Move/copy it under ROOT/checkpoints or ./checkpoints.")

    ptms = [s.strip() for s in args.ptms.split(",") if s.strip()]
    ptm_dim = int(args.ptm_dim)

    # Resolve PTM columns & dataset
    col_map = resolve_ptm_columns(str(MMS_CSV), ptms)
    ds = PTMDataset(str(MMS_CSV), ptms, col_map)
    print(f"[mms] rows={len(ds)} | real={(ds.df['label']==0).sum()} | fake={(ds.df['label']==1).sum()} | speakers={ds.df['speaker_id'].astype(str).nunique()}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} | cuda_available={torch.cuda.is_available()} | gpu={(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')}")

    # Loader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = MoEModel(ptms=ptms, in_dim_each=ptm_dim,
                     expert_hidden=512, expert_drop=0.2,
                     gate_hidden=64, gate_drop=0.1).to(device)

    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state["model"])

    # Inference
    scores, labels = compute_scores_and_labels(model, loader, device)

    # Metrics
    tp, fp, tn, fn, acc05 = confusion(scores, labels, 0.5)
    eer, thr_eer = eer_from_scores(scores, labels)
    tp_e, fp_e, tn_e, fn_e, acc_e = confusion(scores, labels, thr_eer)
    thr_best, acc_best = best_accuracy_threshold(scores, labels)
    tp_b, fp_b, tn_b, fn_b, _ = confusion(scores, labels, thr_best)

    print("\n[MMS TEST RESULTS]")
    print(f"  Acc@0.5         = {acc05:.4f}  (TP={tp} FP={fp} TN={tn} FN={fn})")
    print(f"  EER             = {eer:.4f}  thr@eer={thr_eer:.6f}")
    print(f"  Acc@thr@eer     = {acc_e:.4f}  (TP={tp_e} FP={fp_e} TN={tn_e} FN={fn_e})")
    print(f"  BestAcc (diag)  = {acc_best:.4f}  thr={thr_best:.6f}  (TP={tp_b} FP={fp_b} TN={tn_b} FN={fn_b})")

if __name__ == "__main__":
    main()
