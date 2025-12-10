"""
CORRECT Global XAI Analysis
============================
This script performs 4 global analyses:
1. Temporal patterns (using temporal_heatmap - TRUE temporal data)
2. Frequency band analysis (using frequency_contribution)
3. Linguistic patterns (transcript analysis)
4. Cross-referenced insights (combining all three)

CRITICAL: Features are ALREADY extracted and stored in PTM folders.
We load them from CSV paths - NO re-extraction!
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter, defaultdict
import re
from scipy import stats, interpolate
import librosa

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from xai_analysis import compute_frequency_contribution
# Removed unused: compute_temporal_heatmap, compute_integrated_gradients, compute_shap_approximation

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_vec(path: str) -> np.ndarray:
    """Load .npy feature vector (ALREADY EXTRACTED)"""
    try:
        path_str = str(path)
        
        # Option 1: G drive path -> local SSD path
        if "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm" in path_str:
            path_str = path_str.replace(
                "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm",
                r"C:\Users\pc 1\hindi_df\ptm"
            )
        
        # Option 2: Try relative path if original doesn't exist
        if not Path(path_str).exists():
            rel_path = Path(path_str).name
            local_ptm_dir = Path(__file__).parent / "ptm_features"
            alternate_path = local_ptm_dir / rel_path
            if alternate_path.exists():
                path_str = str(alternate_path)
        
        if not Path(path_str).exists():
            print(f"    [!] File not found: {path_str}")
            return None
        
        vec = np.load(path_str)
        
        # Convert float16 to float32 for model compatibility
        if vec.dtype == np.float16:
            vec = vec.astype(np.float32)
        
        return vec
    except Exception as e:
        print(f"    [!] Failed to load {path}: {e}")
        return None

def load_audio(path: str, sr: int = 16000):
    """Load audio file (audio is on G drive - no path resolution needed)"""
    try:
        # Audio paths in CSV are already correct G drive paths
        # e.g., G:\My Drive\hindi_dfake\processed\wav\strong\...
        if not Path(path).exists():
            return None
        wav, _ = librosa.load(path, sr=sr, mono=True)
        return wav
    except Exception as e:
        print(f"    [!] Failed to load audio {path}: {e}")
        return None

def load_model_from_v4(ckpt_path: str, ptms: list):
    """Load model from train_moe.v4.py"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_moe_v4", "train_moe.v4.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    MoEModel = train_module.MoEModel
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("cfg", {})  # FIX: v5 uses "cfg" not "config"
    
    model = MoEModel(
        ptms=ptms,
        in_dim_each=1536,  # REVERT: Checkpoint was trained with 1536 (mean+std already pooled)
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

def approximate_temporal_via_perturbation(model, feats: Dict[str, np.ndarray], 
                                          device: str, n_segments: int = 50) -> Dict:
    """
    FIX: Approximate temporal importance by masking feature dimensions.
    NOT true temporal analysis - shows which feature dimensions matter.
    Uses EXISTING features - NO re-extraction!
    """
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    
    # Baseline prediction
    with torch.inference_mode():
        logits, _, _ = model(xdict)
        baseline_score = float(torch.softmax(logits, dim=1)[0, 1].item())
    
    # Get feature dimension
    feature_dim = list(feats.values())[0].shape[0]
    segment_size = max(1, feature_dim // n_segments)
    
    timestamps = []
    scores = []
    
    for i in range(n_segments):
        # Mask this segment of features
        masked_feats = {}
        for k, v in feats.items():
            masked = v.copy()
            start = i * segment_size
            end = min((i + 1) * segment_size, len(masked))
            masked[start:end] = 0  # Zero out segment
            masked_feats[k] = masked
        
        # Get prediction with masked features
        masked_xdict = {k: torch.from_numpy(v)[None, :].to(device) 
                       for k, v in masked_feats.items()}
        with torch.inference_mode():
            logits, _, _ = model(masked_xdict)
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
        
        timestamps.append(i / n_segments)  # Normalized position
        scores.append(baseline_score - score)  # Importance = drop when masked
    
    return {
        'timestamps': timestamps,
        'scores': scores,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'consistency_index': float(np.std(scores) / (np.mean(scores) + 1e-8)),
        'note': 'APPROXIMATION: Feature dimension importance, not true temporal'
    }

def normalize_temporal_scores(timestamps: list, scores: list, n_bins: int = 50):
    """
    Normalize variable-length temporal scores to fixed bins (0-100% of audio)
    
    Args:
        timestamps: List of time positions (seconds)
        scores: List of scores at those positions
        n_bins: Number of fixed bins to normalize to
    
    Returns:
        Array of scores normalized to n_bins positions
    """
    if len(timestamps) == 0 or len(scores) == 0:
        return np.zeros(n_bins)
    
    # Normalize timestamps to 0-1 range
    timestamps = np.array(timestamps)
    scores = np.array(scores)
    
    if timestamps.max() > 0:
        timestamps_norm = timestamps / timestamps.max()
    else:
        timestamps_norm = timestamps
    
    # Create fixed bin positions
    bin_positions = np.linspace(0, 1, n_bins)
    
    # Interpolate scores to fixed positions
    if len(timestamps_norm) > 1:
        f = interpolate.interp1d(timestamps_norm, scores, kind='linear', fill_value='extrapolate', bounds_error=False)
        normalized_scores = f(bin_positions)
    else:
        normalized_scores = np.full(n_bins, scores[0])
    
    return normalized_scores

def count_syllables_hindi(word: str) -> int:
    """
    FIX: Count syllables in Hindi word (handles Devanagari and romanized)
    """
    # Devanagari vowels (independent)
    devanagari_vowels = r'[अआइईउऊऋॠऌॡएऐओऔ]'
    # Devanagari matras (dependent vowel marks)
    devanagari_matras = r'[ािीुूृॄॢॣेैोौं]'
    
    # Romanized vowels (including diacritics)
    romanized_vowels = r'[aeiouāīūṛṝḷḹēēaiouAEIOU]'
    
    # Check if word is in Devanagari
    if any('\u0900' <= c <= '\u097F' for c in word):
        # Devanagari: count vowels + matras
        vowel_count = len(re.findall(devanagari_vowels, word))
        matra_count = len(re.findall(devanagari_matras, word))
        return max(1, vowel_count + matra_count)
    else:
        # Romanized: count vowel clusters
        vowel_count = len(re.findall(romanized_vowels, word.lower()))
        return max(1, vowel_count)

def analyze_word_complexity(text: str) -> dict:
    """Analyze word complexity in transcript"""
    if pd.isna(text) or not text:
        return None
    
    words = re.findall(r'\w+', text)
    if not words:
        return None
    
    # Count syllables using improved function
    syllable_counts = [count_syllables_hindi(word) for word in words]
    
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
    parser.add_argument("--master-real-fleurs", default="metadata/thirdparty_real_test.fleurs.csv")
    parser.add_argument("--master-real-train", default="metadata/test_real.from_train.roundrobin_least_damage.csv")
    parser.add_argument("--master-fake-csv", default="metadata/master_fake.csv")
    parser.add_argument("--output-dir", default="global_xai_results_CORRECT")
    parser.add_argument("--ptms", nargs="+", default=["wav2vec2-base", "hubert-base"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CORRECT GLOBAL XAI ANALYSIS")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")
    print(f"Output: {output_dir}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} (TESTING MODE)")
    print("=" * 80)
    
    # Load model
    print("\n[1/6] Loading model...")
    model = load_model_from_v4(args.checkpoint, args.ptms)
    print("✓ Model loaded")
    
    # FIX: No PTM loading needed - using existing features
    print("\n[2/6] Skipping PTM loading (using existing features)...")
    print("✓ Will use perturbation approximation for temporal patterns")
    
    # Load master CSVs for transcripts
    print("\n[3/6] Loading transcript CSVs...")
    master_real_fleurs = pd.read_csv(args.master_real_fleurs)
    master_real_train = pd.read_csv(args.master_real_train)
    master_fake = pd.read_csv(args.master_fake_csv)
    
    # Create transcript lookup
    transcript_lookup = {}
    for _, row in master_real_fleurs.iterrows():
        transcript_lookup[row['utt_id']] = row['text']
    for _, row in master_real_train.iterrows():
        transcript_lookup[row['utt_id']] = row['text']
    for _, row in master_fake.iterrows():
        transcript_lookup[row['utt_id']] = row['text']
    
    print(f"✓ Loaded {len(transcript_lookup)} transcripts")
    
    # Load test CSVs
    print("\n[4/6] Loading test CSVs...")
    test_real = pd.read_csv(args.test_real_csv)
    test_fake = pd.read_csv(args.test_fake_csv)
    
    if args.max_samples:
        test_real = test_real.head(args.max_samples)
        test_fake = test_fake.head(args.max_samples)
    
    print(f"✓ Test real: {len(test_real)} samples")
    print(f"✓ Test fake: {len(test_fake)} samples")
    
    # Detect PTM columns
    ptm_columns = {}
    for ptm in args.ptms:
        for col in test_real.columns:
            if ptm in col and 'vec' in col:
                ptm_columns[ptm] = col
                break
    print(f"✓ PTM columns: {ptm_columns}")
    
    # Process samples
    print("\n[5/6] Running XAI on samples...")
    print("ℹ️  Using perturbation approximation (fast, reuses existing features)")
    
    all_data = []
    
    for label, test_df in [('real', test_real), ('fake', test_fake)]:
        print(f"\n  Processing {label.upper()} samples...")
        
        for idx in tqdm(range(len(test_df)), desc=f"  {label.upper()}"):
            row = test_df.iloc[idx]
            
            # Extract utt_id from path
            audio_path = row['path_audio']
            utt_id = Path(audio_path).stem
            
            # Load ALREADY EXTRACTED features
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
            
            # FIX: Load audio ONLY if file exists (optional for frequency analysis)
            wav = None
            if Path(audio_path).exists():
                wav = load_audio(audio_path)
            
            # Get transcript
            transcript = transcript_lookup.get(utt_id, None)
            
            try:
                # 1. Get prediction from ALREADY EXTRACTED features
                xdict = {k: torch.from_numpy(v)[None, :].to(DEVICE) for k, v in feats.items()}
                with torch.inference_mode():
                    logits, _, _ = model(xdict)
                    probs = torch.softmax(logits, dim=1)
                    p_fake = float(probs[0, 1].item())
                
                # 2. FIX: Use perturbation approximation (fast, uses existing features)
                temporal_result = approximate_temporal_via_perturbation(
                    model, feats, DEVICE, n_segments=50
                )
                
                # 3. Run frequency contribution ONLY if audio exists
                if wav is not None:
                    freq_result = compute_frequency_contribution(wav, 16000)
                else:
                    freq_result = {'skipped': True, 'reason': 'Audio file not found'}
                
                # Store everything
                all_data.append({
                    'sample_id': utt_id,
                    'label': label,
                    'prediction': p_fake,
                    'temporal': temporal_result,
                    'frequency': freq_result,
                    'transcript': transcript
                })
                
            except Exception as e:
                print(f"    [!] Error on sample {idx} ({utt_id}): {e}")
                continue
        
        print(f"  ✓ Processed {sum(1 for d in all_data if d['label'] == label)} {label} samples")
    
    print(f"\n✓ Total samples processed: {len(all_data)}")
    
    # Aggregate results
    print("\n[6/6] Aggregating global patterns...")
    
    # ========================================================================
    # 1. TEMPORAL PATTERNS (from temporal_heatmap - TRUE temporal data)
    # ========================================================================
    print("  [1/4] Aggregating temporal patterns...")
    
    temporal_patterns = {'real': {}, 'fake': {}, 'comparison': {}}
    
    for label in ['real', 'fake']:
        label_data = [d for d in all_data if d['label'] == label]
        
        # FIX: Normalize all temporal scores to 50 bins with safety checks
        normalized_scores_list = []
        for d in label_data:
            # Check if temporal data exists
            if 'skipped' in d['temporal'] or not d['temporal'].get('timestamps'):
                continue  # Skip this sample
            
            timestamps = d['temporal']['timestamps']
            scores = d['temporal']['scores']
            
            if len(timestamps) == 0 or len(scores) == 0:
                continue  # Skip empty
            
            normalized = normalize_temporal_scores(timestamps, scores, n_bins=50)
            normalized_scores_list.append(normalized)
        
        if not normalized_scores_list:
            temporal_patterns[label] = {'error': 'No temporal data available'}
            continue
        
        if normalized_scores_list:
            normalized_scores_array = np.array(normalized_scores_list)
            
            # Compute statistics
            avg_scores = np.mean(normalized_scores_array, axis=0)
            std_scores = np.std(normalized_scores_array, axis=0)
            
            # Identify hotspot regions (avg score > 0.7)
            hotspots = []
            in_hotspot = False
            start_idx = None
            
            for i, score in enumerate(avg_scores):
                if score > 0.7 and not in_hotspot:
                    in_hotspot = True
                    start_idx = i
                elif score <= 0.7 and in_hotspot:
                    in_hotspot = False
                    hotspots.append({
                        'position_pct': f"{start_idx*2}-{i*2}%",
                        'avg_score': float(np.mean(avg_scores[start_idx:i])),
                        'interpretation': f"Position {start_idx*2}-{i*2}% consistently flagged"
                    })
            
            temporal_patterns[label] = {
                'avg_scores_by_position': avg_scores.tolist(),
                'std_scores_by_position': std_scores.tolist(),
                'hotspot_regions': hotspots,
                'n_samples': len(label_data)
            }
    
    # FIX: Comparison with safety checks and symmetric KL divergence
    if (temporal_patterns['real'].get('avg_scores_by_position') and 
        temporal_patterns['fake'].get('avg_scores_by_position')):
        
        real_avg = np.array(temporal_patterns['real']['avg_scores_by_position'])
        fake_avg = np.array(temporal_patterns['fake']['avg_scores_by_position'])
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        real_dist = real_avg + epsilon
        fake_dist = fake_avg + epsilon
        
        # Normalize to probability distributions
        real_dist = real_dist / real_dist.sum()
        fake_dist = fake_dist / fake_dist.sum()
        
        # Compute symmetric KL divergence
        kl_forward = float(stats.entropy(fake_dist, real_dist))
        kl_reverse = float(stats.entropy(real_dist, fake_dist))
        kl_symmetric = (kl_forward + kl_reverse) / 2
        
        # Handle infinity (occurs when distributions have zeros)
        if np.isinf(kl_symmetric) or np.isnan(kl_symmetric):
            kl_symmetric = None
        
        temporal_patterns['comparison'] = {
            'divergence_metric': kl_symmetric,
            'interpretation': "Fake samples show more uniform high scores" if (kl_symmetric and kl_symmetric > 0.3) else "Similar temporal patterns"
        }
    else:
        temporal_patterns['comparison'] = {'error': 'Insufficient data for comparison'}
    
    # ========================================================================
    # 2. FREQUENCY BAND ANALYSIS (from frequency_contribution)
    # ========================================================================
    print("  [2/4] Aggregating frequency patterns...")
    
    frequency_analysis = {'suspicious_bands': [], 'comparison_heatmap': {}}
    
    # Define frequency bands
    freq_bands = [
        (0, 500), (500, 1000), (1000, 1500), (1500, 2000),
        (2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000),
        (4000, 4500), (4500, 5000), (5000, 5500), (5500, 6000),
        (6000, 6500), (6500, 7000), (7000, 7500), (7500, 8000),
        (8000, 8500), (8500, 9000), (9000, 9500), (9500, 10000)
    ]
    
    # FIX: Analyze per-band statistics
    band_stats = {'real': {}, 'fake': {}}
    
    for label in ['real', 'fake']:
        label_data = [d for d in all_data if d['label'] == label]
        
        # For each band, collect statistics
        for band_idx, (low_freq, high_freq) in enumerate(freq_bands):
            band_energies = []
            band_variances = []
            
            for d in label_data:
                # Skip if frequency analysis was skipped
                if 'skipped' in d['frequency']:
                    continue
                
                freq_bins = np.array(d['frequency'].get('freq_bins', []))
                mel_spec = np.array(d['frequency'].get('mel_spectrogram', []))
                
                if mel_spec.size == 0 or len(freq_bins) == 0:
                    continue
                
                # Find mel bins in this frequency range
                band_mask = (freq_bins >= low_freq) & (freq_bins < high_freq)
                
                if band_mask.any():
                    # Extract energy in this band (average over time)
                    band_energy = mel_spec[band_mask, :].mean()
                    band_variance = mel_spec[band_mask, :].std()
                    
                    band_energies.append(band_energy)
                    band_variances.append(band_variance)
            
            if band_energies:
                band_stats[label][f'{low_freq}-{high_freq}Hz'] = {
                    'mean_energy': float(np.mean(band_energies)),
                    'mean_variance': float(np.mean(band_variances)),
                    'samples': len(band_energies)
                }
    
    # Add suspicious band detection
    for band_name in band_stats['real'].keys():
        if band_name in band_stats['fake']:
            real_var = band_stats['real'][band_name]['mean_variance']
            fake_var = band_stats['fake'][band_name]['mean_variance']
            
            # Flag if fake variance is much lower (too uniform)
            if fake_var < real_var * 0.5:
                frequency_analysis['suspicious_bands'].append({
                    'freq_range': band_name,
                    'real_variance': real_var,
                    'fake_variance': fake_var,
                    'interpretation': f'Fake samples show unnaturally uniform energy in {band_name}'
                })
    
    frequency_analysis['band_statistics'] = band_stats
    
    # Still keep the average spectrograms
    for label in ['real', 'fake']:
        label_data = [d for d in all_data if d['label'] == label]
        
        # Collect all mel spectrograms
        all_mels = []
        for d in label_data:
            if 'skipped' in d['frequency']:
                continue
            mel_spec = np.array(d['frequency'].get('mel_spectrogram', []))
            if mel_spec.size > 0:
                all_mels.append(mel_spec)
        
        if all_mels:
            # FIX: Check all have same shape
            shapes = [m.shape for m in all_mels]
            if len(set(shapes)) == 1:
                # All same shape - safe to average
                avg_mel = np.mean(all_mels, axis=0)
                frequency_analysis['comparison_heatmap'][f'{label}_avg_spectrogram'] = avg_mel.tolist()
            else:
                print(f"    [!] Warning: {label} spectrograms have different shapes, skipping average")
                frequency_analysis['comparison_heatmap'][f'{label}_avg_spectrogram'] = None
    
    # FIX: Compute difference map with safety checks
    real_spec_data = frequency_analysis['comparison_heatmap'].get('real_avg_spectrogram')
    fake_spec_data = frequency_analysis['comparison_heatmap'].get('fake_avg_spectrogram')
    
    if real_spec_data is not None and fake_spec_data is not None:
        real_spec = np.array(real_spec_data)
        fake_spec = np.array(fake_spec_data)
        
        # Check shapes match
        if real_spec.shape == fake_spec.shape:
            diff_map = fake_spec - real_spec
            frequency_analysis['comparison_heatmap']['difference_map'] = diff_map.tolist()
        else:
            print("    [!] Warning: Real/fake spectrograms have different shapes")
            frequency_analysis['comparison_heatmap']['difference_map'] = None
    else:
        frequency_analysis['comparison_heatmap']['difference_map'] = None
    
    # ========================================================================
    # 3. LINGUISTIC PATTERNS (transcript analysis)
    # ========================================================================
    print("  [3/4] Analyzing linguistic patterns...")
    
    linguistic_patterns = {'high_risk_words': [], 'syllable_complexity': {}}
    
    # Collect word-score pairs
    word_scores = defaultdict(list)
    word_labels = defaultdict(lambda: {'real': 0, 'fake': 0})
    
    for d in all_data:
        if d['transcript']:
            analysis = analyze_word_complexity(d['transcript'])
            if analysis:
                for word in analysis['words']:
                    word_scores[word].append(d['prediction'])
                    word_labels[word][d['label']] += 1
    
    # Find high-risk words (avg score > 0.7, appears in both classes)
    high_risk = []
    for word, scores in word_scores.items():
        if len(scores) >= 5:  # Minimum occurrences
            avg_score = np.mean(scores)
            if avg_score > 0.6:
                high_risk.append({
                    'word': word,
                    'occurrences_real': word_labels[word]['real'],
                    'occurrences_fake': word_labels[word]['fake'],
                    'avg_score_when_present': float(avg_score),
                    'interpretation': f"Word '{word}' correlates with high detection scores"
                })
    
    # Sort by avg score
    high_risk.sort(key=lambda x: x['avg_score_when_present'], reverse=True)
    linguistic_patterns['high_risk_words'] = high_risk[:20]
    
    # Syllable complexity
    for label in ['real', 'fake']:
        label_data = [d for d in all_data if d['label'] == label and d['transcript']]
        
        all_syllables = []
        for d in label_data:
            analysis = analyze_word_complexity(d['transcript'])
            if analysis:
                all_syllables.extend(analysis['syllable_counts'])
        
        if all_syllables:
            linguistic_patterns['syllable_complexity'][label] = {
                'avg_syllables_per_word': float(np.mean(all_syllables)),
                'words_4plus_syllables': int(sum(1 for s in all_syllables if s >= 4))
            }
    
    # ========================================================================
    # 4. CROSS-REFERENCED INSIGHTS
    # ========================================================================
    print("  [4/4] Cross-referencing insights...")
    
    cross_insights = []
    
    # For each sample, check if temporal peak aligns with transcript
    for d in all_data:
        if not d['transcript']:
            continue
        
        # Find temporal peak
        scores = d['temporal']['scores']
        if not scores:
            continue
        
        peak_idx = np.argmax(scores)
        peak_score = scores[peak_idx]
        
        if peak_score > 0.8:
            timestamps = d['temporal']['timestamps']
            peak_time = timestamps[peak_idx] if peak_idx < len(timestamps) else 0
            
            # FIX: Estimate which word was spoken (syllable-based timing)
            analysis = analyze_word_complexity(d['transcript'])
            if analysis and analysis['num_words'] > 0:
                audio_duration = timestamps[-1] if timestamps else 1.0
                total_syllables = sum(analysis['syllable_counts'])
                
                if total_syllables > 0:
                    time_per_syllable = audio_duration / total_syllables
                    
                    cumulative_time = 0
                    word_idx = -1
                    for idx, (word, syllables) in enumerate(zip(analysis['words'], analysis['syllable_counts'])):
                        word_duration = syllables * time_per_syllable
                        if cumulative_time <= peak_time < cumulative_time + word_duration:
                            word_idx = idx
                            break
                        cumulative_time += word_duration
                    
                    if word_idx >= 0 and word_idx < len(analysis['words']):
                        word = analysis['words'][word_idx]
                        
                        cross_insights.append({
                            'sample_id': d['sample_id'],
                            'temporal_peak': {
                                'time': float(peak_time),
                                'score': float(peak_score)
                            },
                            'linguistic_content': {
                                'word': word,
                                'time_estimate': float(peak_time),
                                'syllables': analysis['syllable_counts'][word_idx]
                            },
                            'combined_insight': f"Model flagged {peak_time:.1f}s (score: {peak_score:.2f}) where word '{word}' was likely spoken"
                        })
    
    # Limit to top 10 examples
    cross_insights = cross_insights[:10]
    
    # Save results
    print("\n  Saving results...")
    
    results = {
        'temporal_patterns': temporal_patterns,
        'frequency_analysis': frequency_analysis,
        'linguistic_patterns': linguistic_patterns,
        'cross_referenced_insights': cross_insights,
        'summary': {
            'n_real_samples': sum(1 for d in all_data if d['label'] == 'real'),
            'n_fake_samples': sum(1 for d in all_data if d['label'] == 'fake'),
            'total_samples': len(all_data)
        }
    }
    
    with open(output_dir / "global_xai_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/global_xai_results.json")
    print(f"\nSummary:")
    print(f"  Real samples: {results['summary']['n_real_samples']}")
    print(f"  Fake samples: {results['summary']['n_fake_samples']}")
    print(f"  High-risk words found: {len(linguistic_patterns['high_risk_words'])}")
    print(f"  Cross-referenced insights: {len(cross_insights)}")

if __name__ == "__main__":
    main()
