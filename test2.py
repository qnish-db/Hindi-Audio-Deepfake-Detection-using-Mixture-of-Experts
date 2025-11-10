# viz_mms_eer_beauty.py
# Full MMS test inference + EER + beautiful, paper-ready plots (PNG + PDF).
# Run: python viz_mms_eer_beauty.py

from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn

# ================== USER CONFIG ==================
ROOT   = Path(r"G:\My Drive\hindi_dfake")
CSV    = ROOT / "metadata" / "tests" / "test_mms.strong.ptm2.csv"
from pathlib import Path
CKPT = Path(__file__).resolve().parent / "checkpoints/moe_ptm2_simpleNN_v1_best.pt"
CALIB  = ROOT / "metadata" / "runs" / "mms_calibration.json"   # optional reference line

PTMS        = ["wav2vec2-base", "hubert-base"]
PTM_DIM     = 1536
BATCH_SIZE  = 512
NUM_WORKERS = 0

SAVE_SCORES_CSV = True
SCORES_CSV      = ROOT / "metadata" / "runs" / "mms_scores.csv"

SAVE_PNG = True
SAVE_PDF = True
OUT_PNG  = ROOT / "metadata" / "runs" / "mms_eer_plot.png"
OUT_PDF  = ROOT / "metadata" / "runs" / "mms_eer_plot.pdf"

SHOW_BASELINE_05    = True
SHOW_CALIBRATED_THR = True
# =================================================

# ---------- Aesthetics: clean, “insanely good” ----------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.0,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.alpha": 0.25,
})

def _resolve_ptm_columns(csv_path: Path, ptm_list):
    df = pd.read_csv(csv_path, nrows=50)
    cols = list(df.columns)
    def norm(s): return s.lower().replace("-", "").replace("_", "")
    npy_like = []
    for c in cols:
        if df[c].dtype == object:
            vals = df[c].dropna().astype(str)
            if not vals.empty and (vals.str.endswith(".npy").mean() > 0.8):
                npy_like.append(c)
    mapping = {}
    for ptm in ptm_list:
        tgt = norm(ptm)
        cand = [c for c in npy_like if tgt in norm(c)]
        if len(cand) == 1:
            mapping[ptm] = cand[0]
        elif len(cand) > 1:
            cand.sort(key=len)
            mapping[ptm] = cand[0]
        else:
            fall = [c for c in npy_like if c not in mapping.values()]
            if not fall:
                raise ValueError(f"Cannot find .npy column for PTM '{ptm}'. Found cols: {cols}")
            mapping[ptm] = fall[0]
    return mapping

def _load_vec(p: str) -> np.ndarray:
    v = np.load(p)
    return v.astype(np.float32, copy=False) if v.dtype != np.float32 else v

# --------- MoE (same as training) ---------
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
    def forward(self, x_concat): return torch.softmax(self.net(x_concat), dim=1)

class MoEModel(nn.Module):
    def __init__(self, ptms, in_dim_each=1536, expert_hidden=512, expert_drop=0.2,
                 gate_hidden=64, gate_drop=0.1):
        super().__init__()
        self.ptms = ptms
        self.experts = nn.ModuleDict({
            p: SmallExpert(in_dim_each, expert_hidden, expert_drop) for p in ptms
        })
        self.gate = TinyGate(in_dim_concat=in_dim_each*len(ptms),
                             hidden=gate_hidden, drop=gate_drop, n_experts=len(ptms))
    def forward(self, x_dict):
        xs = [x_dict[p] for p in self.ptms]
        x_cat = torch.cat(xs, dim=1)
        w = self.gate(x_cat)  # (B,E)
        logits_e = torch.stack([self.experts[p](x_dict[p]) for p in self.ptms], dim=1)  # (B,E,2)
        logits = (w.unsqueeze(-1) * logits_e).sum(dim=1)  # (B,2)
        return logits

def load_checkpoint(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("cfg", {})
    model = MoEModel(
        ptms=cfg.get("ptms", PTMS),
        in_dim_each=cfg.get("ptm_dim", PTM_DIM),
        expert_hidden=cfg.get("expert_hidden", 512),
        expert_drop=cfg.get("expert_dropout", 0.2),
        gate_hidden=cfg.get("gate_hidden", 64),
        gate_drop=cfg.get("gate_dropout", 0.1),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    best_val_eer = state.get("best_val_eer", None)
    return model, cfg, best_val_eer

@torch.no_grad()
def infer_scores(csv_path, ptms, ptm_cols, model, device, batch_size=512):
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(int).values
    utt_id = df["utt_id"].astype(str).values if "utt_id" in df.columns else np.array([f"id_{i}" for i in range(len(df))])
    paths = {p: df[ptm_cols[p]].astype(str).values for p in ptms}
    N = len(df)
    scores = np.zeros(N, dtype=np.float32)
    for i in range(0, N, batch_size):
        sl = slice(i, min(i+batch_size, N))
        x_dict = {}
        for p in ptms:
            vecs = [_load_vec(vp) for vp in paths[p][sl]]
            x = torch.from_numpy(np.stack(vecs, axis=0))
            x_dict[p] = x.to(device, non_blocking=True)
        logits = model(x_dict)
        probs = torch.softmax(logits, dim=1)[:, 1]
        scores[sl] = probs.detach().cpu().numpy().astype(np.float32)
    return utt_id, scores, labels

def roc_from_scores(scores, labels):
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    P = max(1, (y == 1).sum()); N = max(1, (y == 0).sum())
    tpr, fpr = [], []
    tp = fp = 0
    for i in range(len(s)):
        if y[i] == 1: tp += 1
        else: fp += 1
        if i == len(s)-1 or s[i+1] != s[i]:
            tpr.append(tp / P); fpr.append(fp / N)
    return np.array(fpr), np.array(tpr)

def eer_from_scores(scores, labels):
    order = np.argsort(-scores)
    s = scores[order]; y = labels[order]
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0: return 0.5, 0.5
    tp = fp = 0; fn = P; tn = N
    best_eer = 1.0; thr_at = s[0] if len(s) else 0.0
    prev = np.inf
    for i in range(len(s)):
        sc, lab = s[i], y[i]
        if sc != prev:
            fpr = fp / N; fnr = fn / P
            eer = (fpr + fnr) / 2.0
            if abs(fpr - fnr) < abs(2*best_eer - 1):  # tighter
                best_eer = eer; thr_at = prev if prev != np.inf else sc
            prev = sc
        if lab == 1: tp += 1; fn -= 1
        else: fp += 1; tn -= 1
    fpr = fp / N; fnr = fn / P
    eer = (fpr + fnr) / 2.0
    if abs(fpr - fnr) < abs(2*best_eer - 1):
        best_eer = eer; thr_at = s[-1]
    return float(best_eer), float(thr_at)

def confusion(labels, pred):
    TP = int(((pred==1)&(labels==1)).sum())
    FP = int(((pred==1)&(labels==0)).sum())
    TN = int(((pred==0)&(labels==0)).sum())
    FN = int(((pred==0)&(labels==1)).sum())
    return TP, FP, TN, FN

def pretty_axes(ax):
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    print("[paths]")
    print("  ROOT :", ROOT)
    print("  CSV  :", CSV, "exists=", CSV.exists())
    print("  CKPT :", CKPT, "exists=", CKPT.exists())
    if not CSV.exists():  raise FileNotFoundError(f"CSV not found: {CSV}")
    if not CKPT.exists(): raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    ptm_cols = _resolve_ptm_columns(CSV, PTMS)
    print("[ptm columns]", " | ".join([f"{k} -> {v}" for k, v in ptm_cols.items()]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} | cuda_available={torch.cuda.is_available()} | "
          f"gpu={(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')}")

    model, train_cfg, best_val_eer = load_checkpoint(CKPT, device)
    if best_val_eer is not None:
        print(f"[ckpt] best_val_eer recorded during training: {best_val_eer:.4f}")

    cal_thr = None
    if SHOW_CALIBRATED_THR and CALIB.exists():
        try:
            with open(CALIB, "r", encoding="utf-8") as f:
                cal_thr = float(json.load(f).get("calibrated_thr", 0.06))
            print(f"[calibration] using reference calibrated_thr={cal_thr:.6f}")
        except Exception:
            cal_thr = None

    # Inference
    utt_id, scores, labels = infer_scores(CSV, PTMS, ptm_cols, model, device, BATCH_SIZE)
    real = scores[labels == 0]; fake = scores[labels == 1]
    print(f"[counts] real={len(real)} fake={len(fake)} total={len(scores)}")

    # EER and operating points
    eer, thr_eer = eer_from_scores(scores, labels)
    pred_eer = (scores >= thr_eer).astype(int)
    acc_eer  = (pred_eer == labels).mean()

    pred_05  = (scores >= 0.5).astype(int)
    acc_05   = (pred_05 == labels).mean()

    acc_cal = None; pred_cal = None
    if cal_thr is not None:
        pred_cal = (scores >= cal_thr).astype(int)
        acc_cal  = (pred_cal == labels).mean()

    TP_e, FP_e, TN_e, FN_e = confusion(labels, pred_eer)
    TP_5, FP_5, TN_5, FN_5 = confusion(labels, pred_05)
    if pred_cal is not None:
        TP_c, FP_c, TN_c, FN_c = confusion(labels, pred_cal)

    print("\n[METRICS @ MMS TEST]")
    print(f"  EER={eer:.4f} @ thr={thr_eer:.6f} | Acc@EER={acc_eer:.4f}")
    print(f"  Acc@0.5={acc_05:.4f}")
    if acc_cal is not None:
        print(f"  Acc@calibrated({cal_thr:.6f})={acc_cal:.4f}")
    print("  Conf@EER: TP={} FP={} TN={} FN={}".format(TP_e, FP_e, TN_e, FN_e))

    # Export per-file scores (optional)
    if SAVE_SCORES_CSV:
        out_df = pd.DataFrame({
            "utt_id": utt_id,
            "label": labels,
            "score": scores,
            "pred_at_eer": pred_eer
        })
        SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(SCORES_CSV, index=False)
        print(f"[saved] scores → {SCORES_CSV}")

    # ROC / DET
    fpr, tpr = roc_from_scores(scores, labels)
    fnr = 1.0 - tpr
    idx_eer = int(np.argmin(np.abs(fpr - fnr)))

    # ---------- PLOTS ----------
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.0, 1.0], hspace=0.28)

    # A) Score Distributions
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 70)
    # Step-filled for “publication vibe”
    ax1.hist(real, bins=bins, density=True, alpha=0.35, label="Real", edgecolor="none")
    ax1.hist(fake, bins=bins, density=True, alpha=0.35, label="Fake", edgecolor="none")
    ax1.plot([], [], alpha=0)  # spacing hack for layout
    # verticals
    ax1.axvline(thr_eer, linestyle="--", linewidth=2.0, label=f"EER thr={thr_eer:.3f}")
    if SHOW_BASELINE_05:
        ax1.axvline(0.5, linestyle=":", linewidth=1.8, label="thr=0.5")
    if cal_thr is not None:
        ax1.axvline(cal_thr, linestyle="-.", linewidth=1.8, label=f"cal thr={cal_thr:.3f}")
    ax1.set_title("MMS Test — Score Distributions (P(fake))")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Density")
    pretty_axes(ax1)
    ax1.legend(loc="upper right", frameon=False)

    # Add metric box
    text = (f"EER = {eer:.4f} @ thr={thr_eer:.3f}\n"
            f"Acc@EER = {acc_eer:.3f}\n"
            f"Acc@0.5 = {acc_05:.3f}" + (f"\nAcc@cal = {acc_cal:.3f}" if acc_cal is not None else ""))
    ax1.text(0.02, 0.98, text, transform=ax1.transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc=(1,1,1,0.9), ec=(0,0,0,0.15)))

    # B) DET (FPR vs FNR)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(fpr, fnr, label="DET", zorder=1)
    ax2.plot([fpr[idx_eer]], [fnr[idx_eer]], "o", markersize=6, zorder=2, label=f"EER ≈ {eer:.3f}")
    ax2.set_title("DET Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("False Negative Rate")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    pretty_axes(ax2)
    ax2.legend(loc="upper right", frameon=False)

    # C) ROC
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(fpr, tpr, label="ROC", zorder=1)
    ax3.plot([fpr[idx_eer]], [tpr[idx_eer]], "o", markersize=6, zorder=2, label=f"EER point")
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    pretty_axes(ax3)
    ax3.legend(loc="lower right", frameon=False)

    # Tight and save
    fig.tight_layout()
    if SAVE_PNG:
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
        print(f"[saved] {OUT_PNG}")
    if SAVE_PDF:
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PDF, bbox_inches="tight")
        print(f"[saved] {OUT_PDF}")

    plt.show()

if __name__ == "__main__":
    main()
