"""
Global XAI Analysis Script
===========================
Runs XAI (Integrated Gradients + SHAP) on entire test set and aggregates patterns.

Outputs:
1. JSON files with aggregated statistics
2. Matplotlib figures (for verification)
3. Data for frontend visualization (Tailwind-based)

Usage:
    python global_xai_analysis.py --checkpoint checkpoints/moe_ptm2_v5_aggressive_best.pt
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import from training script (train_moe.v4.py)
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import model classes and utilities from train_moe.v4
import importlib.util
spec = importlib.util.spec_from_file_location("train_moe_v4", Path(__file__).parent / "train_moe.v4.py")
train_moe_v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_moe_v4)

load_vec = train_moe_v4.load_vec
PTMDataset = train_moe_v4.PTMDataset
MoEModel = train_moe_v4.MoEModel
SE1D = train_moe_v4.SE1D
ImprovedExpert = train_moe_v4.ImprovedExpert
TinyGate = train_moe_v4.TinyGate
amp_ctx = train_moe_v4.amp_ctx

# Import XAI methods
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from xai_advanced import compute_integrated_gradients, compute_shap_approximation

# =========================
# CONFIG
# =========================
ROOT = Path(r"G:\My Drive\hindi_dfake")
META = ROOT / "metadata"
OUTPUT_DIR = Path(__file__).parent / "global_xai_results"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# HELPER: Load Model
# =========================
def load_model(ckpt_path: str, ptms: List[str], ptm_dim: int = 1536) -> MoEModel:
    """Load trained MoE model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    
    model = MoEModel(
        ptms=ptms,
        in_dim_each=ptm_dim,
        expert_bottleneck=cfg.get("expert_bottleneck", 768),
        expert_drop=cfg.get("expert_dropout", 0.3),
        gate_hidden=cfg.get("gate_hidden", 64),
        gate_drop=cfg.get("gate_dropout", 0.15),
        use_batchnorm=cfg.get("use_batchnorm", True),
        use_se=cfg.get("use_se", False),
        simple_gate=cfg.get("simple_gate", True),
        stochastic_depth=cfg.get("stochastic_depth", 0.6),
        use_fusion=cfg.get("use_fusion", True),
        fusion_dropout=cfg.get("fusion_dropout", 0.5)
    ).to(DEVICE)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[✓] Loaded model from {ckpt_path}")
    return model

# =========================
# HELPER: Resolve PTM Columns
# =========================
def resolve_ptm_columns(csv_path: str, ptms: List[str]) -> Dict[str, str]:
    """Find PTM feature columns in CSV."""
    df = pd.read_csv(csv_path, nrows=1)
    cols = {}
    for ptm in ptms:
        candidates = [c for c in df.columns if ptm in c and "vec" in c.lower()]
        if not candidates:
            raise ValueError(f"No column found for PTM '{ptm}' in {csv_path}")
        cols[ptm] = candidates[0]
    return cols

# =========================
# STEP 1: Run XAI on All Test Samples
# =========================
def run_xai_on_test_set(
    model: MoEModel,
    test_real_csv: str,
    test_fake_csv: str,
    ptms: List[str],
    ptm_columns: Dict[str, str],
    batch_size: int = 128
) -> Dict:
    """
    Run Integrated Gradients and SHAP on all test samples.
    
    Returns:
        {
            'real': {
                'ig_attributions': List[np.ndarray],  # Per sample
                'shap_attributions': List[np.ndarray],
                'predictions': List[float],
                'sample_ids': List[str]
            },
            'fake': { ... same structure ... }
        }
    """
    print("\n[STEP 1] Running XAI on test set...")
    
    results = {'real': {}, 'fake': {}}
    
    for label, csv_path in [('real', test_real_csv), ('fake', test_fake_csv)]:
        print(f"\n  Processing {label.upper()} samples from {csv_path}")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"  Total rows: {len(df)}")
        
        ig_attributions = []
        shap_attributions = []
        predictions = []
        sample_ids = []
        
        # Process in batches for efficiency
        for idx in tqdm(range(len(df)), desc=f"  {label.upper()}", unit="sample"):
            row = df.iloc[idx]
            
            # Load features
            feats = {}
            valid = True
            for ptm in ptms:
                col = ptm_columns[ptm]
                vec_path = row[col]
                vec = load_vec(vec_path)
                if vec is None:
                    valid = False
                    break
                feats[ptm] = vec
            
            if not valid:
                continue
            
            # Run XAI
            try:
                # Integrated Gradients
                ig_result = compute_integrated_gradients(model, feats, DEVICE, steps=50)
                ig_attr = ig_result.get('temporal_attribution', [])
                if not ig_attr:
                    raise ValueError("IG returned empty temporal_attribution")
                ig_attr = np.array(ig_attr)
                
                # SHAP approximation
                shap_result = compute_shap_approximation(model, feats, DEVICE, n_samples=100)
                # SHAP returns 'temporal_importance', not 'temporal_attribution'
                shap_attr = shap_result.get('temporal_importance', shap_result.get('temporal_attribution', []))
                if not shap_attr:
                    raise ValueError("SHAP returned empty temporal_importance")
                shap_attr = np.array(shap_attr)
                
                # Ensure both have same length (pad shorter one)
                max_len = max(len(ig_attr), len(shap_attr))
                if len(ig_attr) < max_len:
                    ig_attr = np.pad(ig_attr, (0, max_len - len(ig_attr)))
                if len(shap_attr) < max_len:
                    shap_attr = np.pad(shap_attr, (0, max_len - len(shap_attr)))
                
                # Get prediction
                xdict = {k: torch.from_numpy(v)[None, :].to(DEVICE) for k, v in feats.items()}
                with torch.inference_mode():
                    logits, _, _ = model(xdict)
                    probs = torch.softmax(logits, dim=1)
                    p_fake = float(probs[0, 1].item())
                
                ig_attributions.append(ig_attr)
                shap_attributions.append(shap_attr)
                predictions.append(p_fake)
                sample_ids.append(row.get('utt_id', f"{label}_{idx}"))
                
            except Exception as e:
                print(f"    [!] Error on sample {idx}: {e}")
                continue
        
        results[label] = {
            'ig_attributions': ig_attributions,
            'shap_attributions': shap_attributions,
            'predictions': predictions,
            'sample_ids': sample_ids
        }
        
        print(f"  ✓ Processed {len(ig_attributions)} {label} samples")
    
    return results

# =========================
# STEP 2: Aggregate Patterns
# =========================
def aggregate_patterns(results: Dict) -> Dict:
    """
    Aggregate XAI results to find global patterns.
    
    Returns:
        {
            'real': {
                'ig_mean': np.ndarray,
                'ig_std': np.ndarray,
                'shap_mean': np.ndarray,
                'shap_std': np.ndarray,
                'avg_prediction': float
            },
            'fake': { ... same ... },
            'discriminative': {
                'ig_diff': np.ndarray,  # fake - real
                'shap_diff': np.ndarray,
                'top_features_ig': List[Tuple[int, float]],  # (feature_idx, importance)
                'top_features_shap': List[Tuple[int, float]]
            }
        }
    """
    print("\n[STEP 2] Aggregating patterns...")
    
    aggregated = {}
    
    for label in ['real', 'fake']:
        ig_attrs = np.array(results[label]['ig_attributions'])  # Shape: (n_samples, n_features)
        shap_attrs = np.array(results[label]['shap_attributions'])
        preds = np.array(results[label]['predictions'])
        
        aggregated[label] = {
            'ig_mean': ig_attrs.mean(axis=0),
            'ig_std': ig_attrs.std(axis=0),
            'ig_median': np.median(ig_attrs, axis=0),
            'ig_percentile_25': np.percentile(ig_attrs, 25, axis=0),
            'ig_percentile_75': np.percentile(ig_attrs, 75, axis=0),
            'shap_mean': shap_attrs.mean(axis=0),
            'shap_std': shap_attrs.std(axis=0),
            'shap_median': np.median(shap_attrs, axis=0),
            'avg_prediction': float(preds.mean()),
            'std_prediction': float(preds.std()),
            'n_samples': len(preds)
        }
        
        print(f"  {label.upper()}: n={len(preds)}, avg_pred={preds.mean():.4f}")
    
    # Discriminative patterns (fake - real)
    ig_diff = aggregated['fake']['ig_mean'] - aggregated['real']['ig_mean']
    shap_diff = aggregated['fake']['shap_mean'] - aggregated['real']['shap_mean']
    
    # Top features (sorted by absolute difference)
    top_k = 20
    top_ig_indices = np.argsort(np.abs(ig_diff))[-top_k:][::-1]
    top_shap_indices = np.argsort(np.abs(shap_diff))[-top_k:][::-1]
    
    aggregated['discriminative'] = {
        'ig_diff': ig_diff,
        'shap_diff': shap_diff,
        'top_features_ig': [(int(i), float(ig_diff[i])) for i in top_ig_indices],
        'top_features_shap': [(int(i), float(shap_diff[i])) for i in top_shap_indices]
    }
    
    print(f"  ✓ Computed discriminative patterns")
    print(f"    Top IG feature: idx={top_ig_indices[0]}, diff={ig_diff[top_ig_indices[0]]:.4f}")
    print(f"    Top SHAP feature: idx={top_shap_indices[0]}, diff={shap_diff[top_shap_indices[0]]:.4f}")
    
    return aggregated

# =========================
# STEP 3: Statistical Significance
# =========================
def compute_statistical_significance(results: Dict, aggregated: Dict) -> Dict:
    """
    Compute t-tests for each feature to find statistically significant differences.
    
    Returns:
        {
            'ig_pvalues': np.ndarray,
            'shap_pvalues': np.ndarray,
            'significant_features_ig': List[int],  # p < 0.01
            'significant_features_shap': List[int]
        }
    """
    print("\n[STEP 3] Computing statistical significance...")
    
    from scipy import stats
    
    ig_real = np.array(results['real']['ig_attributions'])
    ig_fake = np.array(results['fake']['ig_attributions'])
    shap_real = np.array(results['real']['shap_attributions'])
    shap_fake = np.array(results['fake']['shap_attributions'])
    
    n_features = ig_real.shape[1]
    
    ig_pvalues = []
    shap_pvalues = []
    
    for i in range(n_features):
        # t-test for each feature
        _, p_ig = stats.ttest_ind(ig_fake[:, i], ig_real[:, i])
        _, p_shap = stats.ttest_ind(shap_fake[:, i], shap_real[:, i])
        ig_pvalues.append(p_ig)
        shap_pvalues.append(p_shap)
    
    ig_pvalues = np.array(ig_pvalues)
    shap_pvalues = np.array(shap_pvalues)
    
    # Significant features (p < 0.01)
    sig_ig = np.where(ig_pvalues < 0.01)[0].tolist()
    sig_shap = np.where(shap_pvalues < 0.01)[0].tolist()
    
    print(f"  ✓ Significant features (p<0.01): IG={len(sig_ig)}, SHAP={len(sig_shap)}")
    
    return {
        'ig_pvalues': ig_pvalues,
        'shap_pvalues': shap_pvalues,
        'significant_features_ig': sig_ig,
        'significant_features_shap': sig_shap
    }

# =========================
# STEP 4: Save Results
# =========================
def save_results(results: Dict, aggregated: Dict, stats: Dict, output_dir: Path):
    """Save all results to JSON files for frontend."""
    print(f"\n[STEP 4] Saving results to {output_dir}...")
    
    # Convert numpy arrays to lists for JSON serialization
    def to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Save aggregated patterns
    with open(output_dir / "aggregated_patterns.json", "w") as f:
        json.dump(to_json_serializable(aggregated), f, indent=2)
    print("  ✓ Saved aggregated_patterns.json")
    
    # Save statistical significance
    with open(output_dir / "statistical_significance.json", "w") as f:
        json.dump(to_json_serializable(stats), f, indent=2)
    print("  ✓ Saved statistical_significance.json")
    
    # Save summary for frontend
    summary = {
        'n_real_samples': aggregated['real']['n_samples'],
        'n_fake_samples': aggregated['fake']['n_samples'],
        'avg_pred_real': aggregated['real']['avg_prediction'],
        'avg_pred_fake': aggregated['fake']['avg_prediction'],
        'top_discriminative_features_ig': aggregated['discriminative']['top_features_ig'][:10],
        'top_discriminative_features_shap': aggregated['discriminative']['top_features_shap'][:10],
        'n_significant_features_ig': len(stats['significant_features_ig']),
        'n_significant_features_shap': len(stats['significant_features_shap'])
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  ✓ Saved summary.json")

# =========================
# STEP 5: Generate Matplotlib Figures (for verification)
# =========================
def generate_figures(aggregated: Dict, stats: Dict, output_dir: Path):
    """Generate matplotlib figures for verification (not for frontend)."""
    print(f"\n[STEP 5] Generating matplotlib figures...")
    
    # Figure 1: Average IG Attribution (Real vs Fake)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(aggregated['real']['ig_mean'], label='Real', color='green', alpha=0.7)
    axes[0].fill_between(
        range(len(aggregated['real']['ig_mean'])),
        aggregated['real']['ig_mean'] - aggregated['real']['ig_std'],
        aggregated['real']['ig_mean'] + aggregated['real']['ig_std'],
        alpha=0.2, color='green'
    )
    axes[0].set_title("Average IG Attribution - REAL Samples")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Attribution Score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(aggregated['fake']['ig_mean'], label='Fake', color='red', alpha=0.7)
    axes[1].fill_between(
        range(len(aggregated['fake']['ig_mean'])),
        aggregated['fake']['ig_mean'] - aggregated['fake']['ig_std'],
        aggregated['fake']['ig_mean'] + aggregated['fake']['ig_std'],
        alpha=0.2, color='red'
    )
    axes[1].set_title("Average IG Attribution - FAKE Samples")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Attribution Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ig_attribution_comparison.png", dpi=150)
    print("  ✓ Saved ig_attribution_comparison.png")
    plt.close()
    
    # Figure 2: Discriminative Features (Fake - Real)
    fig, ax = plt.subplots(figsize=(12, 6))
    ig_diff = aggregated['discriminative']['ig_diff']
    ax.bar(range(len(ig_diff)), ig_diff, color=['red' if x > 0 else 'green' for x in ig_diff], alpha=0.7)
    ax.set_title("Discriminative Features (Fake - Real) - IG")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Attribution Difference")
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "discriminative_features_ig.png", dpi=150)
    print("  ✓ Saved discriminative_features_ig.png")
    plt.close()
    
    # Figure 3: Top 20 Discriminative Features
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = aggregated['discriminative']['top_features_ig'][:20]
    indices = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    colors = ['red' if v > 0 else 'green' for v in values]
    
    ax.barh(range(len(indices)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f"Feature {i}" for i in indices])
    ax.set_xlabel("Attribution Difference (Fake - Real)")
    ax.set_title("Top 20 Discriminative Features - IG")
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / "top20_discriminative_ig.png", dpi=150)
    print("  ✓ Saved top20_discriminative_ig.png")
    plt.close()
    
    print(f"  ✓ All figures saved to {output_dir}")

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Global XAI Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-real-csv", type=str, default=str(META / "fs_test_real.labeled.csv"))
    parser.add_argument("--test-fake-csv", type=str, default=str(META / "fs_test_fake_mms.labeled.csv"))
    parser.add_argument("--ptms", type=str, default="wav2vec2-base,hubert-base")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    
    # Parse PTMs
    ptms = [s.strip() for s in args.ptms.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("GLOBAL XAI ANALYSIS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Real: {args.test_real_csv}")
    print(f"Test Fake: {args.test_fake_csv}")
    print(f"PTMs: {ptms}")
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    # Load model
    model = load_model(args.checkpoint, ptms)
    
    # Resolve PTM columns
    ptm_columns = resolve_ptm_columns(args.test_real_csv, ptms)
    print(f"PTM columns: {ptm_columns}")
    
    # Run XAI on test set
    results = run_xai_on_test_set(
        model, args.test_real_csv, args.test_fake_csv, 
        ptms, ptm_columns
    )
    
    # Aggregate patterns
    aggregated = aggregate_patterns(results)
    
    # Statistical significance
    stats = compute_statistical_significance(results, aggregated)
    
    # Save results
    save_results(results, aggregated, stats, output_dir)
    
    # Generate figures
    generate_figures(aggregated, stats, output_dir)
    
    print("\n" + "="*60)
    print("✓ GLOBAL XAI ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Check matplotlib figures for verification")
    print("2. Use JSON files for frontend visualization")
    print("3. Implement Tailwind-based charts in frontend")

if __name__ == "__main__":
    main()
