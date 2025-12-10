"""
PROPER Global XAI Analysis
- Feature importance (IG)
- Expert contributions (SHAP)
- Transcript analysis (word patterns)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
import re

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from xai_advanced import compute_integrated_gradients, compute_shap_approximation

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_vec(path: str) -> np.ndarray:
    """Load .npy feature vector"""
    try:
        return np.load(path)
    except:
        return None

def load_model_from_v4(ckpt_path: str, ptms: list):
    """Load model from train_moe.v4.py"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_moe_v4", "train_moe.v4.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    MoEModel = train_module.MoEModel
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    
    model = MoEModel(
        ptms=ptms,
        in_dim_each=1536,
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
    return model

def analyze_word_complexity(text: str) -> dict:
    """Analyze word complexity in transcript"""
    if pd.isna(text) or not text:
        return None
    
    words = re.findall(r'\w+', text.lower())
    if not words:
        return None
    
    # Count syllables (rough approximation for Hindi)
    syllable_counts = []
    for word in words:
        # Simple heuristic: vowels = syllables
        vowels = len(re.findall(r'[aeiouāīūṛṝḷḹēōṁḥ]', word))
        syllable_counts.append(max(1, vowels))
    
    return {
        'num_words': len(words),
        'avg_syllables': np.mean(syllable_counts),
        'max_syllables': max(syllable_counts),
        'words': words,
        'syllable_counts': syllable_counts
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-real-csv", default="metadata/fs_test_real.labeled.csv")
    parser.add_argument("--test-fake-csv", default="metadata/fs_test_fake_mms.labeled.csv")
    parser.add_argument("--master-real-csv", default="metadata/master_real.csv")
    parser.add_argument("--master-fake-csv", default="metadata/master_fake.csv")
    parser.add_argument("--output-dir", default="global_xai_proper_results")
    parser.add_argument("--ptms", nargs="+", default=["wav2vec2-base", "hubert-base"])
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PROPER GLOBAL XAI ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model_from_v4(args.checkpoint, args.ptms)
    print("✓ Model loaded")
    
    # Load master CSVs for transcripts
    print("\n[2/5] Loading master CSVs for transcripts...")
    master_real = pd.read_csv(args.master_real_csv)
    master_fake = pd.read_csv(args.master_fake_csv)
    print(f"✓ Master real: {len(master_real)} rows")
    print(f"✓ Master fake: {len(master_fake)} rows")
    
    # Load test CSVs
    print("\n[3/5] Loading test CSVs...")
    test_real = pd.read_csv(args.test_real_csv)
    test_fake = pd.read_csv(args.test_fake_csv)
    print(f"✓ Test real: {len(test_real)} rows")
    print(f"✓ Test fake: {len(test_fake)} rows")
    
    # Detect PTM columns
    ptm_columns = {}
    for ptm in args.ptms:
        for col in test_real.columns:
            if ptm in col and 'vec' in col:
                ptm_columns[ptm] = col
                break
    print(f"✓ PTM columns: {ptm_columns}")
    
    # Process samples
    print("\n[4/5] Running XAI on samples...")
    
    results = {
        'real': {'ig_features': [], 'shap_experts': [], 'predictions': [], 'transcripts': [], 'utt_ids': []},
        'fake': {'ig_features': [], 'shap_experts': [], 'predictions': [], 'transcripts': [], 'utt_ids': []}
    }
    
    for label, test_df, master_df in [
        ('real', test_real, master_real),
        ('fake', test_fake, master_fake)
    ]:
        print(f"\n  Processing {label.upper()} samples...")
        
        for idx in tqdm(range(len(test_df)), desc=f"  {label.upper()}"):
            row = test_df.iloc[idx]
            
            # Load features
            feats = {}
            valid = True
            for ptm in args.ptms:
                col = ptm_columns.get(ptm)
                if not col:
                    valid = False
                    break
                vec_path = row[col]
                vec = load_vec(vec_path)
                if vec is None:
                    valid = False
                    break
                feats[ptm] = vec
            
            if not valid:
                continue
            
            # Get transcript
            utt_id = row.get('utt_id', '')
            transcript = None
            if utt_id and not pd.isna(utt_id):
                match = master_df[master_df['utt_id'] == utt_id]
                if len(match) > 0:
                    transcript = match.iloc[0].get('text', '')
            
            try:
                # Run IG
                ig_result = compute_integrated_gradients(model, feats, DEVICE, steps=50)
                ig_features = ig_result.get('attribution_per_expert', {})
                
                # Run SHAP
                shap_result = compute_shap_approximation(model, feats, DEVICE, n_samples=100)
                shap_experts = shap_result.get('expert_contributions', {})
                
                # Get prediction
                xdict = {k: torch.from_numpy(v)[None, :].to(DEVICE) for k, v in feats.items()}
                with torch.inference_mode():
                    logits, _, _ = model(xdict)
                    probs = torch.softmax(logits, dim=1)
                    p_fake = float(probs[0, 1].item())
                
                results[label]['ig_features'].append(ig_features)
                results[label]['shap_experts'].append(shap_experts)
                results[label]['predictions'].append(p_fake)
                results[label]['transcripts'].append(transcript)
                results[label]['utt_ids'].append(utt_id)
                
            except Exception as e:
                print(f"    [!] Error on sample {idx}: {e}")
                continue
        
        print(f"  ✓ Processed {len(results[label]['predictions'])} {label} samples")
    
    # Aggregate results
    print("\n[5/5] Aggregating results...")
    
    # 1. Feature importance aggregation
    print("  - Aggregating feature importance...")
    feature_importance = {'real': {}, 'fake': {}}
    for label in ['real', 'fake']:
        for ptm in args.ptms:
            all_features = []
            for sample_features in results[label]['ig_features']:
                if ptm in sample_features:
                    all_features.append(sample_features[ptm])
            
            if all_features:
                # Pad to same length
                max_len = max(len(f) for f in all_features)
                padded = [f + [0.0] * (max_len - len(f)) for f in all_features]
                feature_importance[label][ptm] = np.mean(padded, axis=0).tolist()
    
    # 2. Expert contributions aggregation
    print("  - Aggregating expert contributions...")
    expert_contributions = {'real': {}, 'fake': {}}
    for label in ['real', 'fake']:
        for ptm in args.ptms:
            contributions = [s.get(ptm, 0.0) for s in results[label]['shap_experts']]
            expert_contributions[label][ptm] = {
                'mean': float(np.mean(contributions)),
                'std': float(np.std(contributions)),
                'min': float(np.min(contributions)),
                'max': float(np.max(contributions))
            }
    
    # 3. Transcript analysis
    print("  - Analyzing transcripts...")
    transcript_analysis = {'real': {}, 'fake': {}}
    for label in ['real', 'fake']:
        all_words = []
        syllable_distribution = []
        word_complexity = []
        
        for transcript in results[label]['transcripts']:
            if transcript and not pd.isna(transcript):
                analysis = analyze_word_complexity(transcript)
                if analysis:
                    all_words.extend(analysis['words'])
                    syllable_distribution.extend(analysis['syllable_counts'])
                    word_complexity.append(analysis['avg_syllables'])
        
        # Word frequency
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(50)
        
        transcript_analysis[label] = {
            'total_words': len(all_words),
            'unique_words': len(set(all_words)),
            'top_words': [{'word': w, 'count': c} for w, c in top_words],
            'avg_syllables_per_word': float(np.mean(syllable_distribution)) if syllable_distribution else 0,
            'syllable_distribution': {
                '1': sum(1 for s in syllable_distribution if s == 1),
                '2': sum(1 for s in syllable_distribution if s == 2),
                '3': sum(1 for s in syllable_distribution if s == 3),
                '4+': sum(1 for s in syllable_distribution if s >= 4)
            }
        }
    
    # 4. Summary stats
    summary = {
        'n_real_samples': len(results['real']['predictions']),
        'n_fake_samples': len(results['fake']['predictions']),
        'avg_pred_real': float(np.mean(results['real']['predictions'])),
        'avg_pred_fake': float(np.mean(results['fake']['predictions'])),
        'std_pred_real': float(np.std(results['real']['predictions'])),
        'std_pred_fake': float(np.std(results['fake']['predictions']))
    }
    
    # Save results
    print("\n  Saving results...")
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    
    with open(output_dir / "expert_contributions.json", "w") as f:
        json.dump(expert_contributions, f, indent=2)
    
    with open(output_dir / "transcript_analysis.json", "w") as f:
        json.dump(transcript_analysis, f, indent=2)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"\nSummary:")
    print(f"  Real samples: {summary['n_real_samples']} (avg pred: {summary['avg_pred_real']:.3f})")
    print(f"  Fake samples: {summary['n_fake_samples']} (avg pred: {summary['avg_pred_fake']:.3f})")

if __name__ == "__main__":
    main()
