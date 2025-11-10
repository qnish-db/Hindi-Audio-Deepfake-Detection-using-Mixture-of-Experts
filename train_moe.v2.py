# train_moe.v2.py
# Mixture-of-Experts over cached PTM vectors (2+ PTMs), with auxiliary losses.
# TQDM logging, idempotent, AMP-enabled, early-stopping on Val EER.
# Logs EER + Accuracy and writes a small run manifest.

import os, json, math, time, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from tqdm import tqdm
from contextlib import nullcontext

# -------------------------
# ROOT & metadata paths (cleaner fix)
# -------------------------
ROOT = Path(r"G:\My Drive\hindi_dfake")   # <-- change here if needed
META = ROOT / "metadata"

# -------------------------
# Test-set map (add new test manifests here)
# -------------------------
TEST_CSV_MAP = {
    "core": META / "split_test.ptm2.csv",                     # original default
    "edge": META / "tests" / "test_edge.strong.ptm2.csv",     # optional
    "mms":  META / "tests" / "test_mms.strong.ptm2.csv",      # MMS test you built
}

# -------------------------
# Config (edit here or via CLI)
# -------------------------
DEFAULTS = dict(
    seed=1337,
    # absolute paths via ROOT/META
    train_csv=str(META / "split_train.ptm2.csv"),
    val_csv=str(META / "split_val.ptm2.csv"),
    # --- default to MMS test set ---
    test_csv=str(META / "tests" / "test_mms.strong.ptm2.csv"),
    test_set="mms",                               # default switch = MMS
    # --------------------------------
    ptms=["wav2vec2-base", "hubert-base"],        # add more later
    ptm_dim=1536,                                 # each PTM vector dim
    batch_size=512,
    balanced_sampler=True,                        # set False to disable
    num_workers=4,
    expert_hidden=512,
    expert_dropout=0.2,
    gate_hidden=64,
    gate_dropout=0.1,
    lr=1e-3,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    warmup_ratio=0.05,
    cosine_final_lr=1e-4,
    max_epochs=30,
    patience=5,
    label_smoothing=0.05,
    aux_loss_lambda=0.2,                          # set 0.0 to disable aux loss
    grad_clip=1.0,
    amp=True,
    ckpt_dir="checkpoints",
    run_name="moe_ptm2_simpleNN_v1",
    quiet=False,
)

# -------------------------
# Test-set resolver (tiny switcher)
# -------------------------
def resolve_test_csv(cfg):
    ts = str(cfg.get("test_set", "")).lower().strip()
    if ts and ts != "custom":
        p = TEST_CSV_MAP.get(ts, None)
        if p is not None and p.exists():
            cfg["test_csv"] = str(p)
            print(f"[test] using test_set='{ts}' -> {cfg['test_csv']}")
        else:
            print(f"[warn] test_set='{ts}' not found on disk; keeping test_csv='{cfg['test_csv']}'")
    else:
        # 'custom' means: leave test_csv exactly as user provided
        print(f"[test] using custom test_csv='{cfg['test_csv']}'")
    return cfg

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_vec(path: str) -> np.ndarray:
    v = np.load(path)
    if v.dtype != np.float32:
        v = v.astype(np.float32, copy=False)
    return v

def amp_ctx(device: str, enabled: bool):
    if enabled and device == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()

@torch.no_grad()
def compute_scores_and_labels(model, loader, device, return_gate=False):
    model.eval()
    all_scores, all_labels, gate_collect = [], [], []
    for batch in loader:
        # MOVE DICT TENSORS TO DEVICE (fix for CPU/CUDA mismatch)
        xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = batch["y"].to(device, non_blocking=True)

        with amp_ctx(device, enabled=True):  # safe & lightweight in eval
            logits, _, gate_w = model(xs)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)

        all_scores.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        if return_gate:
            gate_collect.append(gate_w.cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    if return_gate:
        gates = np.concatenate(gate_collect, axis=0)
        return scores, labels, gates
    return scores, labels

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    correct = (pred == labels).sum().item()
    return correct / labels.numel()

def eer_from_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-scores)
    scores = scores[order]; labels = labels[order]
    P = (labels == 1).sum(); N = (labels == 0).sum()
    if P == 0 or N == 0:
        return 0.5, 0.0
    tp = fp = 0; fn = P; tn = N
    best_diff = 1.0; eer = 1.0
    thr_at_eer = scores[0] if len(scores) else 0.0
    prev_s = np.inf
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

def make_warmup_cosine(total_steps: int, warmup_ratio: float, base_lr: float, final_lr: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return final_lr / base_lr + (1 - final_lr / base_lr) * cosine
    return lr_lambda

# -------------------------
# Column resolver (auto-detect)
# -------------------------
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

# -------------------------
# Data
# -------------------------
class PTMDataset(Dataset):
    def __init__(self, csv_path: str, ptm_list: List[str], ptm_columns: Dict[str, str]):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.ptms = ptm_list
        self.ptm_cols = ptm_columns
        for ptm in self.ptms:
            col = self.ptm_cols[ptm]
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' for PTM '{ptm}' in {csv_path}")
        if "label" not in self.df.columns:
            raise ValueError(f"Missing 'label' column in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xs = {}
        for ptm in self.ptms:
            p = row[self.ptm_cols[ptm]]
            v = load_vec(p)
            xs[ptm] = torch.from_numpy(v)
        y = int(row["label"])
        return {"x": xs, "y": torch.tensor(y, dtype=torch.long)}

def collate_fn(batch_list):
    ptm_names = list(batch_list[0]["x"].keys())
    xs = {ptm: torch.stack([b["x"][ptm] for b in batch_list], dim=0) for ptm in ptm_names}
    y = torch.stack([b["y"] for b in batch_list], dim=0)
    return {"x": xs, "y": y}

def make_loader(csv_path, ptms, ptm_cols, batch_size, shuffle, num_workers, balanced=False):
    ds = PTMDataset(csv_path, ptms, ptm_cols)
    if balanced:
        labels = ds.df["label"].values
        class_counts = np.bincount(labels, minlength=2).astype(np.float32)
        class_weights = class_counts.sum() / (2.0 * class_counts + 1e-9)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return ds, loader

# -------------------------
# Model
# -------------------------
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

    def forward(self, x):
        return self.net(x)

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
        w = torch.softmax(w, dim=1)
        return w

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
            in_dim_concat = in_dim_each * len(ptms),
            hidden = gate_hidden,
            drop = gate_drop,
            n_experts = len(ptms)
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]):
        ptm_order = self.ptms
        xs = [x_dict[ptm] for ptm in ptm_order]
        x_concat = torch.cat(xs, dim=1)
        gate_w = self.gate(x_concat)                     # (B,E)

        expert_logits = []
        for ptm in ptm_order:
            expert_logits.append(self.experts[ptm](x_dict[ptm]))  # (B,2)
        expert_logits = torch.stack(expert_logits, dim=1)         # (B,E,2)

        final_logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)  # (B,2)
        return final_logits, expert_logits, gate_w

# -------------------------
# Training / Eval
# -------------------------
def train_loop(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # >>> Explicit CUDA/Device print <<<
    print(
        "[env]",
        "device=", device,
        "| cuda_available=", torch.cuda.is_available(),
        "| gpu=", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        flush=True
    )
    # <<<

    set_seed(cfg["seed"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # ---- Resolve PTM columns automatically from TRAIN CSV header
    PTM_COLUMNS = resolve_ptm_columns(cfg["train_csv"], cfg["ptms"])

    # Data
    train_ds, train_loader = make_loader(
        cfg["train_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=not cfg["balanced_sampler"], num_workers=cfg["num_workers"],
        balanced=cfg["balanced_sampler"]
    )
    # Reuse the same mapping for val/test (same headers)
    val_ds, val_loader = make_loader(
        cfg["val_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], balanced=False
    )
    test_ds, test_loader = make_loader(
        cfg["test_csv"], cfg["ptms"], PTM_COLUMNS, cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], balanced=False
    )

    # Friendly data summary
    tr = train_ds.df["label"].value_counts().to_dict()
    va = val_ds.df["label"].value_counts().to_dict()
    print(f"[data] train={len(train_ds)} (real={tr.get(0,0)}, fake={tr.get(1,0)}) | "
          f"val={len(val_ds)} (real={va.get(0,0)}, fake={va.get(1,0)}) | test={len(test_ds)}")

    # Model
    model = MoEModel(
        ptms=cfg["ptms"],
        in_dim_each=cfg["ptm_dim"],
        expert_hidden=cfg["expert_hidden"],
        expert_drop=cfg["expert_dropout"],
        gate_hidden=cfg["gate_hidden"],
        gate_drop=cfg["gate_dropout"]
    ).to(device)

    # Optim / sched
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=cfg["betas"]
    )
    total_steps = cfg["max_epochs"] * math.ceil(len(train_ds) / cfg["batch_size"])
    lr_lambda = make_warmup_cosine(total_steps, cfg["warmup_ratio"], cfg["lr"], cfg["cosine_final_lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # Loss
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = GradScaler(enabled=(device == "cuda") and cfg["amp"])

    best_val_eer = 1.0
    epochs_no_improve = 0
    best_ckpt = Path(cfg["ckpt_dir"]) / f"{cfg['run_name']}_best.pt"
    last_ckpt = Path(cfg["ckpt_dir"]) / f"{cfg['run_name']}_last.pt"

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"epoch {epoch:02d}", leave=False)
        for batch in pbar:
            xs = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
            y  = batch["y"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with amp_ctx(device, enabled=cfg["amp"]):
                logits, expert_logits, gate_w = model(xs)
                main_loss = ce(logits, y)

                aux_loss = 0.0
                if cfg["aux_loss_lambda"] > 0.0:
                    B, E, C = expert_logits.shape
                    aux = 0.0
                    for e in range(E):
                        aux += ce(expert_logits[:, e, :], y)
                    aux_loss = (cfg["aux_loss_lambda"] * aux / E)

                loss = main_loss + aux_loss

            scaler.scale(loss).backward()
            if cfg["grad_clip"] is not None and cfg["grad_clip"] > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()

            running_loss += loss.item()
            running_acc  += accuracy_from_logits(logits, y)
            n_batches += 1

            if not cfg["quiet"]:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(running_acc/n_batches):.4f}")

        # ---- Validation
        val_scores, val_labels = compute_scores_and_labels(model, val_loader, device)
        val_eer, _ = eer_from_scores(val_scores, val_labels)

        model.eval()
        v_acc_num = 0; v_acc_den = 0
        with torch.no_grad():
            for batch in val_loader:
                xs = {k: v.to(device) for k, v in batch["x"].items()}
                y  = batch["y"].to(device)
                logits, _, _ = model(xs)
                v_acc_num += (torch.argmax(logits,1) == y).sum().item()
                v_acc_den += y.numel()
        val_acc = v_acc_num / max(1, v_acc_den)

        train_loss = running_loss / max(1, n_batches)
        train_acc  = running_acc / max(1, n_batches)
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_eer {val_eer:.4f} | val_acc {val_acc:.4f}")

        # ---- Early stopping / checkpoint
        improved = val_eer < best_val_eer - 1e-5
        if improved:
            best_val_eer = val_eer
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(),
                        "cfg": cfg,
                        "epoch": epoch,
                        "best_val_eer": best_val_eer}, best_ckpt)
        else:
            epochs_no_improve += 1

        torch.save({"model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "best_val_eer": best_val_eer}, last_ckpt)

        if epochs_no_improve >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}. Best val EER={best_val_eer:.4f}")
            break

    # -----------------
    # Final TEST report (load best)
    # -----------------
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model"])
    test_scores, test_labels = compute_scores_and_labels(model, test_loader, device)
    test_eer, thr = eer_from_scores(test_scores, test_labels)

    model.eval()
    t_acc_num = 0; t_acc_den = 0
    with torch.no_grad():
        for batch in test_loader:
            xs = {k: v.to(device) for k, v in batch["x"].items()}
            y  = batch["y"].to(device)
            logits, _, _ = model(xs)
            t_acc_num += (torch.argmax(logits,1) == y).sum().item()
            t_acc_den += y.numel()
    test_acc = t_acc_num / max(1, t_acc_den)

    print("\n[RESULTS]")
    print(f" Best Val EER : {best_val_eer:.4f}")
    print(f" Test EER     : {test_eer:.4f} (thr@eer={thr:.5f})")
    print(f" Test Acc     : {test_acc:.4f}")

    run_manifest = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_val_eer": float(best_val_eer),
        "test_eer": float(test_eer),
        "thr_at_eer": float(thr),
        "test_acc": float(test_acc),
        "cfg": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in cfg.items()},
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    runs_dir = META / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / f"{cfg['run_name']}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

def parse_args():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        arg = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            ap.add_argument(arg, action="store_true" if not v else "store_false")
        elif isinstance(v, (list, tuple)):
            ap.add_argument(arg, type=str, default=",".join(map(str, v)))
        else:
            ap.add_argument(arg, type=type(v), default=v)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = DEFAULTS.copy()
    for k in cfg.keys():
        v = getattr(args, k)
        if isinstance(DEFAULTS[k], (list, tuple)) and isinstance(v, str):
            if k == "betas":
                parts = v.split(","); cfg[k] = (float(parts[0]), float(parts[1]))
            else:
                cfg[k] = [s.strip() for s in v.split(",") if s.strip()]
        else:
            cfg[k] = v
    if isinstance(cfg["betas"], str):
        parts = cfg["betas"].split(","); cfg["betas"] = (float(parts[0]), float(parts[1]))

    # ---- resolve test set switch (keeps MMS by default) ----
    cfg = resolve_test_csv(cfg)

    set_seed(cfg["seed"])
    train_loop(cfg)

if __name__ == "__main__":
    main()
