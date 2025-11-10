# eval_only.py — evaluate + calibrate using the trained best checkpoint
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- keep these in sync with train script ----
PTMS = ["wav2vec2-base","hubert-base"]
PTM_DIM = 1536
BATCH = 512
NUM_WORKERS = 0  # <= important on Windows to avoid spawn + pagefile issues

def to_int_labels(s):
    if s.dtype == object:
        low = s.astype(str).str.lower()
        mapped = np.where(low.isin(["fake","1"]), 1,
                 np.where(low.isin(["real","0"]), 0, np.nan))
        out = pd.Series(mapped)
    else:
        out = s
    return out.fillna(0).astype(np.int64).values

def resolve_ptm_columns(csv_path, ptm_list):
    dfh = pd.read_csv(csv_path, nrows=200)
    cols = list(dfh.columns)
    def norm(x): return str(x).lower().replace("-","").replace("_","")
    npy_like = []
    for c in cols:
        if dfh[c].dtype == object:
            v = dfh[c].dropna().astype(str)
            if not v.empty and (v.str.endswith(".npy").mean() > 0.7):
                npy_like.append(c)
    col_map = {}
    for p in ptm_list:
        t = norm(p)
        cands = [c for c in npy_like if t in norm(c)]
        if len(cands)==1: col_map[p]=cands[0]
        elif len(cands)>1:
            cands.sort(key=len); col_map[p]=cands[0]
        else:
            fallback = [c for c in npy_like if c not in col_map.values()]
            assert fallback, f"No .npy column for {p}"
            col_map[p]=fallback[0]
    return col_map

# local redirect for features (same rule as your train script)
DRIVE_PREFIX = r"G:\My Drive\hindi_dfake\processed\features\ptm"
LOCAL_PREFIX = r"C:\Users\pc 1\hindi_df\ptm"
MIN_NPY_BYTES = 1024

def load_vec(path):
    if DRIVE_PREFIX in path:
        path = path.replace(DRIVE_PREFIX, LOCAL_PREFIX)
    try:
        if not os.path.isfile(path): return None
        if os.path.getsize(path) < MIN_NPY_BYTES: return None
        v = np.load(path, mmap_mode="r")
        if v.dtype != np.float32:
            v = v.astype(np.float32, copy=False)
        if v.ndim != 1 or v.shape[0] <= 0:
            return None
        return v
    except Exception:
        return None

class PTMDataset(Dataset):
    def __init__(self, csv_path, ptm_list, ptm_cols):
        df = pd.read_csv(csv_path).reset_index(drop=True)
        self.df = df
        self.ptms = ptm_list
        self.ptm_cols = ptm_cols
        self.labels = to_int_labels(df["label"])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        xs = {}
        for p in self.ptms:
            v = load_vec(r[self.ptm_cols[p]])
            if v is None: return None
            xs[p] = torch.from_numpy(v)
        y = int(self.labels[idx])
        return {"x": xs, "y": torch.tensor(y, dtype=torch.long)}

def collate_fn(bl):
    bl = [b for b in bl if b is not None]
    if not bl: return {"x":{}, "y": torch.empty(0, dtype=torch.long)}
    ptm_names = list(bl[0]["x"].keys())
    xs = {p: torch.stack([b["x"][p] for b in bl], 0) for p in ptm_names}
    y = torch.stack([b["y"] for b in bl], 0)
    return {"x": xs, "y": y}

class SE1D(nn.Module):
    def __init__(self, C, reduction=16):
        super().__init__()
        r = max(1, C//reduction)
        self.fc1, self.fc2 = nn.Linear(C, r), nn.Linear(r, C)
        self.act, self.sig = nn.ReLU(inplace=True), nn.Sigmoid()
    def forward(self, x):
        w = self.fc2(self.act(self.fc1(x)))
        return x * self.sig(w)

class ImprovedExpert(nn.Module):
    def __init__(self, in_dim=1536, bottleneck=768, drop=0.2):
        super().__init__()
        self.pre = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, bottleneck),
                                 nn.GELU(), nn.Dropout(drop))
        self.se = SE1D(bottleneck)
        self.mid = nn.Sequential(nn.Linear(bottleneck, bottleneck), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(bottleneck, 2)
    def forward(self, x):
        h = self.pre(x); h = self.se(h); h = h + self.mid(h); return self.head(h)

class TinyGate(nn.Module):
    def __init__(self, in_dim_concat, hidden=64, drop=0.1, n_experts=2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim_concat),
                                 nn.Linear(in_dim_concat, hidden),
                                 nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, n_experts))
    def forward(self, x): return torch.softmax(self.net(x), 1)

class MoEModel(nn.Module):
    def __init__(self, ptms, in_dim_each=1536, expert_bottleneck=768, expert_drop=0.2,
                 gate_hidden=64, gate_drop=0.1):
        super().__init__()
        self.ptms = ptms
        self.experts = nn.ModuleDict({p: ImprovedExpert(in_dim_each, expert_bottleneck, expert_drop)
                                      for p in ptms})
        self.gate = TinyGate(in_dim_each*len(ptms), gate_hidden, gate_drop, len(ptms))
    def forward(self, xdict):
        xs = [xdict[p] for p in self.ptms]
        concat = torch.cat(xs, 1)
        w = self.gate(concat)
        ex = torch.stack([self.experts[p](x) for p,x in zip(self.ptms, xs)], 1)
        return (w.unsqueeze(-1)*ex).sum(1)

@torch.no_grad()
def scores_on_loader(model, loader, device):
    model.eval()
    outs, labs = [], []
    for b in loader:
        if b is None or b["y"].numel()==0: continue
        x = {k:v.to(device) for k,v in b["x"].items()}
        y = b["y"].numpy().astype(np.int32)
        logits = model(x)
        probs = torch.softmax(logits, 1)[:,1].cpu().numpy()
        outs.append(probs); labs.append(y)
    if not outs: return np.zeros((0,),dtype=np.float32), np.zeros((0,),dtype=np.int32)
    return np.concatenate(outs), np.concatenate(labs)

def eer_from_scores(scores, labels):
    if len(scores)==0: return 0.5, 0.5
    o = np.argsort(-scores)
    s, y = scores[o], labels[o]
    P, N = (y==1).sum(), (y==0).sum()
    if P==0 or N==0: return 0.5, 0.5
    tp=fp=0; fn=P; tn=N
    best=1.0; eer=1.0; thr=s[0]; prev=np.inf
    for i in range(len(s)):
        si, yi = s[i], y[i]
        if si!=prev:
            fpr=fp/N; fnr=fn/P; diff=abs(fpr-fnr)
            if diff<best: best=diff; eer=(fpr+fnr)/2.0; thr=prev
            prev=si
        if yi==1: tp+=1; fn-=1
        else: fp+=1; tn-=1
    fpr=fp/N; fnr=fn/P; diff=abs(fpr-fnr)
    if diff<best: eer=(fpr+fnr)/2.0; thr=s[-1]
    return float(eer), float(thr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/moe_ptm2_se768_v3_best.pt")
    ap.add_argument("--test-real-csv", type=str, required=True)
    ap.add_argument("--test-fake-csv", type=str, required=True)
    ap.add_argument("--val-csv", type=str, required=False)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-frac", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    # resolve PTM columns from either CSV
    ref_csv = args.test_real_csv
    ptm_cols = resolve_ptm_columns(ref_csv, PTMS)

    # loaders (num_workers=0 on purpose)
    def make(csvp):
        ds = PTMDataset(csvp, PTMS, ptm_cols)
        return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True, collate_fn=collate_fn)

    dl_real = make(args.test_real_csv)
    dl_fake = make(args.test_fake_csv)

    # model
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg", {})
    model = MoEModel(PTMS, PTM_DIM, int(cfg.get("expert_bottleneck",768)),
                     float(cfg.get("expert_dropout",0.2)),
                     int(cfg.get("gate_hidden",64)),
                     float(cfg.get("gate_dropout",0.1))).to(device)
    model.load_state_dict(ck["model"])

    # scores
    s_r, y_r = scores_on_loader(model, dl_real, device)
    s_f, y_f = scores_on_loader(model, dl_fake, device)
    scores = np.concatenate([s_r, s_f])
    labels = np.concatenate([y_r, np.ones_like(y_f)])

    eer, thr = eer_from_scores(scores, labels)
    acc05 = float(((scores>=0.5).astype(np.int32)==labels).mean())
    acc_eer = float(((scores>=thr).astype(np.int32)==labels).mean())

    print("\n[RESULTS]")
    print(f" Test EER   : {eer:.4f} (thr@eer={thr:.6f})")
    print(f" Acc@0.5    : {acc05:.4f}")
    print(f" Acc@thrEER : {acc_eer:.4f}")

    if args.calibrate:
        rng = np.random.default_rng(1337)
        pos = np.where(labels==1)[0]
        neg = np.where(labels==0)[0]
        npos = max(1, int(len(pos)*args.calib_frac))
        nneg = max(1, int(len(neg)*args.calib_frac))
        dev = set(rng.choice(pos, npos, replace=False).tolist() +
                  rng.choice(neg, nneg, replace=False).tolist())
        hold = [i for i in range(len(labels)) if i not in dev]
        dev_s, dev_y = scores[list(dev)], labels[list(dev)]
        hold_s, hold_y = scores[hold], labels[hold]
        grid = np.unique(np.concatenate([np.unique(dev_s), np.linspace(0,1,401)]))
        best_t, best_a = 0.5, -1.0
        for t in grid:
            a = float(((dev_s>=t).astype(np.int32)==dev_y).mean())
            if a>best_a: best_a, best_t = a, t
        acc_hold = float(((hold_s>=best_t).astype(np.int32)==hold_y).mean())
        print("\n[Calibration]")
        print(f"  dev_frac={args.calib_frac:.2f} dev_n={len(dev)} hold_n={len(hold)}")
        print(f"  best_thr={best_t:.6f} (dev acc={best_a:.4f})")
        print(f"  hold_acc={acc_hold:.4f}")

if __name__ == "__main__":
    main()
