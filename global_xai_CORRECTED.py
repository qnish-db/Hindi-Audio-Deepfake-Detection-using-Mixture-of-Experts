#!/usr/bin/env python3
"""
Global XAI Analysis — CORRECTED VERSION
All critical errors fixed based on code review
"""

import os, sys, argparse, json, warnings, importlib.util
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor, AutoModel
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# Try to import epitran for phoneme analysis
try:
    from epitran import Epitran
    EPITRAN_AVAILABLE = True
    epi = None  # Will be initialized later to avoid encoding issues
except ImportError:
    EPITRAN_AVAILABLE = False
    epi = None
    print("Warning: epitran not installed. Phoneme analysis will be skipped.")
    print("Install with: pip install epitran")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
WAV2VEC2_ID = "facebook/wav2vec2-base"
HUBERT_ID = "facebook/hubert-base-ls960"

# Import modules
sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("train_moe_v4", Path(__file__).parent / "train_moe.v4.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
MoEModel = train_module.MoEModel

spec_xai = importlib.util.spec_from_file_location("xai_analysis", Path(__file__).parent / "backend" / "xai_analysis.py")
xai_module = importlib.util.module_from_spec(spec_xai)
spec_xai.loader.exec_module(xai_module)
compute_temporal_heatmap = xai_module.compute_temporal_heatmap
compute_frequency_contribution = xai_module.compute_frequency_contribution


def load_model_checkpoint(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config", {}))  # FIXED: v5 uses "cfg", v4 uses "config"
    ptms = cfg.get("ptms", ["wav2vec2-base", "hubert-base"])
    
    model = MoEModel(
        ptms=ptms, in_dim_each=1536,
        expert_bottleneck=cfg.get("expert_bottleneck", 768),
        expert_drop=cfg.get("expert_dropout", 0.3),
        gate_hidden=cfg.get("gate_hidden", 64),
        gate_drop=cfg.get("gate_dropout", 0.15),
        use_batchnorm=cfg.get("use_batchnorm", True),
        use_se=cfg.get("use_se", False),
        simple_gate=cfg.get("simple_gate", True),  # FIXED: v5 uses True
        stochastic_depth=cfg.get("stochastic_depth", 0.6),
        use_fusion=cfg.get("use_fusion", True),  # FIXED: v5 uses True
        fusion_dropout=cfg.get("fusion_dropout", 0.5)
    ).to(device)
    
    # FIXED: v5 uses "model" key, not "model_state_dict"
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg, ptms


class PTMFeatureExtractor:
    def __init__(self, device: str):
        self.device = device
        self.wav2vec2_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ID)
        self.wav2vec2_model = AutoModel.from_pretrained(WAV2VEC2_ID, use_safetensors=True).to(device).eval()
        self.hubert_fe = AutoFeatureExtractor.from_pretrained(HUBERT_ID)
        self.hubert_model = AutoModel.from_pretrained(HUBERT_ID, use_safetensors=True).to(device).eval()
    
    @torch.inference_mode()
    def extract_wav2vec2_frames(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        batch = self.wav2vec2_fe(audio, sampling_rate=sr, return_tensors="pt")
        out = self.wav2vec2_model(batch.input_values.to(self.device), output_hidden_states=True, return_dict=True)
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
        return hs[-1][0].cpu().numpy().astype(np.float32)
    
    @torch.inference_mode()
    def extract_hubert_frames(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        batch = self.hubert_fe(audio, sampling_rate=sr, return_tensors="pt")
        out = self.hubert_model(batch.input_values.to(self.device), output_hidden_states=True, return_dict=True)
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
        return hs[-1][0].cpu().numpy().astype(np.float32)
    
    @staticmethod
    def pool_to_1536(frames: np.ndarray) -> np.ndarray:
        mu = frames.mean(axis=0)
        sd = frames.std(axis=0)
        return np.concatenate([mu, sd], axis=0).astype(np.float32)


def count_syllables_hindi(word: str) -> int:
    hindi_vowels = set('अआइईउऊएऐओऔaeiouAEIOU')
    return max(1, sum(1 for c in word if c in hindi_vowels))


def extract_phonemes(word: str) -> List[str]:
    """Extract phonemes from Hindi word using epitran"""
    global epi
    if not EPITRAN_AVAILABLE or epi is None:
        return []
    try:
        # Transliterate to IPA phonemes
        ipa = epi.transliterate(word)
        # Split into individual phonemes (simple split by character for now)
        phonemes = list(ipa)
        return [p for p in phonemes if p.strip()]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv-real", default="metadata/fs_test_real.labeled.csv")  # FIXED: Use same as train_moe.v4.py
    parser.add_argument("--test-csv-fake", default="metadata/fs_test_fake_mms.labeled.csv")  # FIXED: Use same as train_moe.v4.py
    parser.add_argument("--master-real", default="metadata/master_real.csv")
    parser.add_argument("--master-fake", default="metadata/master_fake.csv")
    parser.add_argument("--output-dir", default="global_xai_results_FINAL")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=50)
    args = parser.parse_args()
    
    # Disable TensorFlow to avoid import issues
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['USE_TF'] = '0'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}\nGlobal XAI Analysis - CORRECTED\n{'='*70}")
    print(f"Device: {DEVICE}")
    
    # Initialize epitran if available
    global epi
    if EPITRAN_AVAILABLE:
        print("Note: Epitran has encoding issues on Windows. Phoneme analysis will be skipped.")
        epi = None
        # Uncomment below to try epitran (may fail on Windows):
        # try:
        #     print("Initializing epitran for phoneme analysis...")
        #     import pandas as pd
        #     original_read_csv = pd.read_csv
        #     def read_csv_utf8(*args, **kwargs):
        #         if 'encoding' not in kwargs:
        #             kwargs['encoding'] = 'utf-8'
        #         return original_read_csv(*args, **kwargs)
        #     pd.read_csv = read_csv_utf8
        #     epi = Epitran('hin-Deva')
        #     print("  Epitran initialized successfully")
        # except Exception as e:
        #     print(f"  Warning: Could not initialize epitran: {e}")
        #     epi = None
    
    # Load model
    print("\n[1/5] Loading model...")
    model, cfg, ptms = load_model_checkpoint(args.checkpoint, DEVICE)
    print(f"  Model loaded: {ptms}")
    
    # Load PTM extractors
    print("\n[2/5] Loading PTM feature extractors...")
    ptm_extractor = PTMFeatureExtractor(device=DEVICE)
    
    # Feature extractor function for compute_temporal_heatmap
    def feature_extractor_fn(audio_segment: np.ndarray) -> Dict[str, np.ndarray]:
        w2v_frames = ptm_extractor.extract_wav2vec2_frames(audio_segment, TARGET_SR)
        hub_frames = ptm_extractor.extract_hubert_frames(audio_segment, TARGET_SR)
        return {
            'wav2vec2-base': PTMFeatureExtractor.pool_to_1536(w2v_frames),
            'hubert-base': PTMFeatureExtractor.pool_to_1536(hub_frames)
        }
    
    # Load test data
    print("\n[3/5] Loading test data...")
    test_real = pd.read_csv(args.test_csv_real)
    test_fake = pd.read_csv(args.test_csv_fake)
    if args.max_samples:
        test_real = test_real.head(args.max_samples)
        test_fake = test_fake.head(args.max_samples)
    print(f"  Real: {len(test_real)}, Fake: {len(test_fake)}")
    
    # Load transcripts
    master_real = pd.read_csv(args.master_real)
    master_fake = pd.read_csv(args.master_fake)
    transcript_lookup = {}
    for df in [master_real, master_fake]:
        for _, row in df.iterrows():
            text = row.get("text") or row.get("transcript")
            if pd.notna(text):
                transcript_lookup[row["utt_id"]] = str(text)
    print(f"  Transcripts loaded: {len(transcript_lookup)} entries")
    
    # Debug: Check for transcript match with test data
    test_all = pd.concat([test_real, test_fake])
    matched_transcripts = sum(1 for uid in test_all['utt_id'] if uid in transcript_lookup)
    print(f"  Transcript matches with test data: {matched_transcripts}/{len(test_all)}")
    if matched_transcripts == 0:
        print("  WARNING: No transcript matches found! Cross-reference linguistic analysis will be limited.")
    
    # Initialize results
    temporal_patterns = {
        "real": {"avg_scores_by_position": None, "std_scores_by_position": None, "hotspot_regions": [], "n_samples": 0},
        "fake": {"avg_scores_by_position": None, "std_scores_by_position": None, "hotspot_regions": [], "n_samples": 0},
        "comparison": {}
    }
    frequency_analysis = {"suspicious_bands": [], "comparison_heatmap": {}, "band_statistics": {"real": {}, "fake": {}}}
    linguistic_patterns = {"high_risk_words": [], "syllable_complexity": {"real": {}, "fake": {}}}
    cross_referenced_insights = []
    
    # Process samples
    print("\n[4/5] Processing samples...")
    word_predictions = []
    all_sample_data = []  # Store per-sample data for cross-referencing
    phoneme_predictions = []  # Store (phoneme, score, label) for phoneme analysis
    
    for label, test_df in [("real", test_real), ("fake", test_fake)]:
        print(f"\n  Processing {label}... ({len(test_df)} samples)")
        temporal_scores_list = []
        freq_results = []
        all_syllables = []
        mel_spectrograms = []  # For comparison heatmap
        samples_processed = 0
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {label}"):
            try:
                utt_id = row.get("utt_id", "")  # FIXED: Define utt_id at the top
                audio_path = row.get("path_audio") or row.get("path")
                
                # FIXED: Resolve path for real samples (convert /content/drive/MyDrive to G:\My Drive)
                if audio_path and "/content/drive/MyDrive/" in audio_path:
                    audio_path = audio_path.replace("/content/drive/MyDrive/", "G:\\My Drive\\")
                    audio_path = audio_path.replace("/", "\\")
                
                if not audio_path or not Path(audio_path).exists():
                    continue
                
                audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != TARGET_SR:
                    continue
                
                # Temporal analysis using compute_temporal_heatmap
                temporal_result = compute_temporal_heatmap(
                    model, audio, sr, feature_extractor_fn, DEVICE,
                    window_sec=0.5, hop_sec=0.25
                )
                
                if temporal_result['scores']:
                    scores = np.array(temporal_result['scores'])
                    binned = np.interp(
                        np.linspace(0, len(scores)-1, args.n_bins),
                        np.arange(len(scores)),
                        scores
                    )
                    temporal_scores_list.append(binned)
                    overall_score = temporal_result['mean_score']
                else:
                    overall_score = 0.0
                
                # Frequency analysis
                freq_result = compute_frequency_contribution(audio, sr)
                freq_results.append(freq_result)
                
                # Store mel spectrogram for comparison heatmap
                if 'mel_spectrogram' in freq_result:
                    mel_spectrograms.append(freq_result['mel_spectrogram'])
                
                # Store per-sample data for cross-referencing
                transcript_text = transcript_lookup.get(utt_id, '')
                # DEBUG: Track if transcript is missing
                if not transcript_text:
                    # Try alternative matching (strip spaces, lowercase)
                    alt_utt_id = utt_id.strip().lower()
                    for key in transcript_lookup:
                        if key.strip().lower() == alt_utt_id:
                            transcript_text = transcript_lookup[key]
                            break
                
                sample_data = {
                    'utt_id': utt_id,
                    'label': label,
                    'temporal_scores': binned if temporal_result['scores'] else None,
                    'temporal_timestamps': temporal_result.get('timestamps', []),
                    'freq_result': freq_result,
                    'transcript': transcript_text,
                    'audio_duration': len(audio) / sr,
                    'overall_score': overall_score
                }
                all_sample_data.append(sample_data)
                
                # Linguistic analysis - use the transcript_text we already found (consistent with sample_data)
                if transcript_text:
                    words = transcript_text.split()
                    for word in words:
                        syllables = count_syllables_hindi(word)
                        all_syllables.append(syllables)
                        word_predictions.append((word, overall_score, label))
                        
                        # Extract phonemes if epitran available
                        if EPITRAN_AVAILABLE:
                            phonemes = extract_phonemes(word)
                            for phoneme in phonemes:
                                phoneme_predictions.append((phoneme, overall_score, label))
                
                samples_processed += 1
                
            except Exception as e:
                continue
        
        print(f"  {label}: Successfully processed {samples_processed}/{len(test_df)} samples")
        
        # Aggregate temporal
        if temporal_scores_list:
            arr = np.array(temporal_scores_list)
            avg_scores = arr.mean(axis=0)
            temporal_patterns[label]["avg_scores_by_position"] = avg_scores.tolist()
            temporal_patterns[label]["std_scores_by_position"] = arr.std(axis=0).tolist()
            temporal_patterns[label]["n_samples"] = len(temporal_scores_list)
            
            # HOTSPOT DETECTION
            threshold = 0.6
            hotspots = []
            in_hotspot = False
            hotspot_start = None
            
            for i, score in enumerate(avg_scores):
                if score > threshold and not in_hotspot:
                    hotspot_start = i
                    in_hotspot = True
                elif score <= threshold and in_hotspot:
                    position_pct = f"{int(hotspot_start * 100 / len(avg_scores))}-{int(i * 100 / len(avg_scores))}%"
                    hotspots.append({
                        "position_pct": position_pct,
                        "avg_score": float(np.mean(avg_scores[hotspot_start:i])),
                        "interpretation": "Consistently high detection scores in this region"
                    })
                    in_hotspot = False
            
            # Handle case where hotspot extends to end
            if in_hotspot:
                position_pct = f"{int(hotspot_start * 100 / len(avg_scores))}-100%"
                hotspots.append({
                    "position_pct": position_pct,
                    "avg_score": float(np.mean(avg_scores[hotspot_start:])),
                    "interpretation": "Consistently high detection scores in this region"
                })
            
            temporal_patterns[label]["hotspot_regions"] = hotspots
        
        # Aggregate frequency - BAND-BY-BAND BREAKDOWN
        if freq_results:
            # Overall high-freq stats
            high_freq = [f.get('high_freq_energy', 0.0) for f in freq_results]
            
            # Band-by-band analysis (16 bands from 0-8kHz)
            band_stats = {}
            for band_idx in range(16):
                freq_low = band_idx * 500
                freq_high = (band_idx + 1) * 500
                band_name = f"{freq_low}-{freq_high}Hz"
                
                # Extract energy and variance for this band from all samples
                band_energies = []
                band_variances = []
                
                for freq_res in freq_results:
                    mel_spec = freq_res.get('mel_spectrogram', [])
                    if len(mel_spec) > 0:
                        # Map Hz to mel bins (approximate)
                        n_mels = len(mel_spec)
                        mel_low = band_idx * n_mels // 16
                        mel_high = (band_idx + 1) * n_mels // 16
                        mel_high = max(mel_high, mel_low + 1)
                        
                        band_data = mel_spec[mel_low:mel_high]
                        if len(band_data) > 0:
                            band_energies.append(float(np.mean(band_data)))
                            band_variances.append(float(np.var(band_data)))
                
                if band_energies:
                    band_stats[band_name] = {
                        "mean_energy": float(np.mean(band_energies)),
                        "mean_variance": float(np.mean(band_variances)),
                        "samples": len(band_energies)
                    }
            
            frequency_analysis["band_statistics"][label] = band_stats
            
            # Flag suspicious bands (low variance = unnaturally uniform)
            if label == "fake":
                for band_name, stats in band_stats.items():
                    if stats["mean_variance"] < 5.0:  # Threshold for "too uniform"
                        frequency_analysis["suspicious_bands"].append({
                            "freq_range": band_name,
                            "mean_variance": stats["mean_variance"],
                            "interpretation": f"Unnaturally uniform energy in {band_name} (typical of TTS)"
                        })
        
        # Generate comparison heatmap
        if mel_spectrograms:
            # FIXED: Normalize all spectrograms to same time dimension before averaging
            n_freq_bins = 20
            n_time_bins = 50
            
            # Downsample each spectrogram to fixed size first
            normalized_specs = []
            for mel_spec in mel_spectrograms:
                if len(mel_spec) == 0 or len(mel_spec[0]) == 0:
                    continue
                
                # Downsample this spectrogram to 20x50
                downsampled = np.zeros((n_freq_bins, n_time_bins))
                n_mels = len(mel_spec)
                n_time = len(mel_spec[0])
                
                for i in range(n_freq_bins):
                    for j in range(n_time_bins):
                        freq_start = i * n_mels // n_freq_bins
                        freq_end = (i + 1) * n_mels // n_freq_bins
                        time_start = j * n_time // n_time_bins
                        time_end = (j + 1) * n_time // n_time_bins
                        
                        # Extract region and compute mean
                        region = [mel_spec[f][time_start:time_end] for f in range(freq_start, freq_end)]
                        if region and len(region[0]) > 0:
                            downsampled[i, j] = np.mean([np.mean(row) for row in region if len(row) > 0])
                
                normalized_specs.append(downsampled)
            
            # Now average all normalized spectrograms
            if normalized_specs:
                avg_spec = np.mean(normalized_specs, axis=0)
                frequency_analysis["comparison_heatmap"][f"{label}_avg_spectrogram"] = avg_spec.tolist()
        
        # Linguistic stats
        if all_syllables:
            linguistic_patterns["syllable_complexity"][label] = {
                "avg_syllables_per_word": float(np.mean(all_syllables)),
                "words_4plus_syllables": int(sum(1 for s in all_syllables if s >= 4))
            }
    
    # Word-score correlations
    print("\n  Computing linguistic correlations...")
    word_scores = defaultdict(list)
    for word, score, label in word_predictions:
        word_scores[word].append(score)
    
    word_avg = [(w, np.mean(s), len(s)) for w, s in word_scores.items() if len(s) >= 3]
    word_avg.sort(key=lambda x: x[1], reverse=True)
    
    linguistic_patterns["high_risk_words"] = [
        {"word": w, "avg_score_when_present": float(s), "occurrences": int(c)}
        for w, s, c in word_avg[:20]
    ]
    
    # Phoneme analysis (if epitran available)
    if EPITRAN_AVAILABLE and phoneme_predictions:
        print("  Computing phoneme analysis...")
        phoneme_counts = defaultdict(lambda: {"real": 0, "fake": 0, "real_high_score": 0, "fake_high_score": 0})
        
        for phoneme, score, label in phoneme_predictions:
            phoneme_counts[phoneme][label] += 1
            if score > 0.7:  # High score threshold
                phoneme_counts[phoneme][f"{label}_high_score"] += 1
        
        # Compute mispronunciation rate
        phoneme_analysis = []
        for phoneme, counts in phoneme_counts.items():
            if counts["fake"] >= 5:  # Minimum occurrences
                fake_high_pct = (counts["fake_high_score"] / counts["fake"]) * 100 if counts["fake"] > 0 else 0
                real_high_pct = (counts["real_high_score"] / counts["real"]) * 100 if counts["real"] > 0 else 0
                
                # Only include if significantly different
                if abs(fake_high_pct - real_high_pct) > 10:
                    phoneme_analysis.append({
                        "phoneme": phoneme,
                        "mispronounced_pct_fake": float(fake_high_pct),
                        "mispronounced_pct_real": float(real_high_pct),
                        "occurrences_fake": int(counts["fake"]),
                        "occurrences_real": int(counts["real"]),
                        "interpretation": f"TTS struggles with this phoneme" if fake_high_pct > real_high_pct else "Real audio struggles more"
                    })
        
        # Sort by difference in mispronunciation rate
        phoneme_analysis.sort(key=lambda x: abs(x["mispronounced_pct_fake"] - x["mispronounced_pct_real"]), reverse=True)
        linguistic_patterns["phoneme_analysis"] = phoneme_analysis[:20]  # Top 20
    else:
        linguistic_patterns["phoneme_analysis"] = []
    
    # Temporal comparison
    print("\n[5/5] Computing comparison metrics...")
    if temporal_patterns["real"]["avg_scores_by_position"] and temporal_patterns["fake"]["avg_scores_by_position"]:
        real_dist = np.abs(np.array(temporal_patterns["real"]["avg_scores_by_position"]))
        fake_dist = np.abs(np.array(temporal_patterns["fake"]["avg_scores_by_position"]))
        real_dist /= (real_dist.sum() + 1e-9)
        fake_dist /= (fake_dist.sum() + 1e-9)
        
        kl_div = float(scipy_stats.entropy(fake_dist + 1e-9, real_dist + 1e-9))
        if np.isinf(kl_div) or np.isnan(kl_div):
            kl_div = None
        
        temporal_patterns["comparison"] = {
            "divergence_metric": kl_div,
            "interpretation": "Different temporal patterns" if kl_div and kl_div > 0.3 else "Similar patterns"
        }
    
    # Compute difference heatmap
    if "real_avg_spectrogram" in frequency_analysis["comparison_heatmap"] and "fake_avg_spectrogram" in frequency_analysis["comparison_heatmap"]:
        real_spec = np.array(frequency_analysis["comparison_heatmap"]["real_avg_spectrogram"])
        fake_spec = np.array(frequency_analysis["comparison_heatmap"]["fake_avg_spectrogram"])
        difference = fake_spec - real_spec
        frequency_analysis["comparison_heatmap"]["difference_map"] = difference.tolist()
    
    # CROSS-REFERENCED INSIGHTS - PER-SAMPLE ALIGNMENT
    print("\n  Computing cross-referenced insights...")
    
    temporal_freq_alignments = 0
    temporal_linguistic_alignments = 0
    all_three_alignments = 0
    total_peaks = 0
    samples_with_transcripts = 0
    samples_with_freq_data = 0
    
    for sample_data in all_sample_data[:100]:  # Limit to first 100 for performance
        # FIXED: Check if temporal_scores is None or empty (not numpy array boolean)
        temporal_scores = sample_data['temporal_scores']
        if temporal_scores is None or len(temporal_scores) == 0:
            continue
        
        audio_duration = sample_data['audio_duration']
        transcript = sample_data['transcript']
        words = transcript.split() if transcript else []
        
        # Track data availability
        if words:
            samples_with_transcripts += 1
        
        # Find temporal peaks using dynamic threshold (top 3 above 85th percentile)
        ts = np.array(temporal_scores, dtype=float)
        if ts.size == 0:
            continue
        perc85 = float(np.percentile(ts, 85))
        top_idx = np.argsort(ts)[-3:][::-1]
        peak_indices = [int(i) for i in top_idx if ts[int(i)] >= perc85]
        if not peak_indices:
            # Fallback: take top 3 anyway
            peak_indices = [int(i) for i in top_idx]
        total_peaks += len(peak_indices)
        
        for peak_idx in peak_indices[:3]:  # Top 3 peaks per sample
            # Convert bin index to time
            time_sec = (peak_idx / len(temporal_scores)) * audio_duration
            peak_score = temporal_scores[peak_idx]
            
            # Check frequency anomaly at that time
            freq_result = sample_data['freq_result']
            mel_spec = freq_result.get('mel_spectrogram', [])
            has_freq_anomaly = False
            freq_bands = []
            
            if len(mel_spec) > 0 and len(mel_spec[0]) > 0:
                samples_with_freq_data += 1
                time_idx = int((time_sec / audio_duration) * len(mel_spec[0]))
                time_idx = min(time_idx, len(mel_spec[0]) - 1)
                
                # Extract frequency slice at this time
                time_slice = np.array([row[time_idx] if time_idx < len(row) else 0 for row in mel_spec])
                
                if len(time_slice) > 0:
                    # FIXED: Multiple anomaly criteria (more lenient)
                    slice_variance = np.var(time_slice)
                    slice_mean = np.mean(time_slice)
                    slice_std = np.std(time_slice)
                    
                    # Anomaly if ANY of these conditions met:
                    # 1. Very low variance relative to mean (unnaturally uniform)
                    # 2. Unusually low energy across multiple bands
                    # 3. High standard deviation (erratic energy)
                    
                    is_low_variance = slice_variance < max(1.0, abs(slice_mean) * 0.05)  # < 5% of mean
                    is_low_energy = slice_mean < -50  # Very quiet
                    is_high_std = slice_std > 20  # Erratic
                    
                    if is_low_variance or is_low_energy or is_high_std:
                        has_freq_anomaly = True
                        
                        # Find which bands are anomalous
                        for band_idx in range(min(16, len(time_slice) // 8)):
                            start = band_idx * len(time_slice) // 16
                            end = (band_idx + 1) * len(time_slice) // 16
                            band_mean = np.mean(time_slice[start:end])
                            band_var = np.var(time_slice[start:end])
                            
                            # Flag band if low energy OR low variance
                            if band_mean < -45 or band_var < 2.0:
                                freq_bands.append(f"{band_idx*500}-{(band_idx+1)*500}Hz")
            
            # Estimate word at this time (uniform distribution)
            word = None
            if words:
                word_idx = int((time_sec / audio_duration) * len(words))
                word_idx = min(word_idx, len(words) - 1)
                word = words[word_idx]
            
            # Count alignments
            if has_freq_anomaly:
                temporal_freq_alignments += 1
            if word:
                temporal_linguistic_alignments += 1
            if has_freq_anomaly and word:
                all_three_alignments += 1

            # Add to insights (limit to 10), even if only some modalities present
            if len(cross_referenced_insights) < 10:
                insight_obj = {
                    "sample_id": sample_data.get('utt_id', ''),
                    "temporal_peak": {
                        "time_range": f"{time_sec:.1f}s",
                        "score": float(peak_score)
                    }
                }
                if has_freq_anomaly:
                    insight_obj["frequency_anomaly"] = {
                        "freq_range": ", ".join(freq_bands[:2]) if freq_bands else "",
                        "bands": freq_bands,
                        "interpretation": "Low variance/energy at this time"
                    }
                if word:
                    insight_obj["linguistic_content"] = {
                        "word": word,
                        "estimated_time": f"~{time_sec:.1f}s",
                        "syllables": count_syllables_hindi(word)
                    }
                # Combined insight summary
                parts = [f"Model flagged {time_sec:.1f}s (score: {peak_score:.2f})"]
                if word:
                    parts.append(f"word '{word}'")
                if freq_bands:
                    parts.append(f"freq bands {', '.join(freq_bands[:2])}")
                insight_obj["combined_insight"] = "; ".join(parts)
                
                cross_referenced_insights.append(insight_obj)
    
    # Add alignment statistics with debug info
    print(f"\n  Cross-reference stats:")
    print(f"    Total peaks analyzed: {total_peaks}")
    print(f"    Samples with transcripts: {samples_with_transcripts}")
    print(f"    Samples with freq data: {samples_with_freq_data}")
    print(f"    Temporal-Freq alignments: {temporal_freq_alignments}")
    print(f"    Temporal-Linguistic alignments: {temporal_linguistic_alignments}")
    print(f"    All-three alignments: {all_three_alignments}")
    
    if total_peaks > 0:
        cross_referenced_insights.append({
            "pattern_type": "alignment_statistics",
            "temporal_freq_alignment": float(temporal_freq_alignments / total_peaks),
            "temporal_linguistic_alignment": float(temporal_linguistic_alignments / total_peaks),
            "all_three_alignment": float(all_three_alignments / total_peaks),
            "description": f"{int(all_three_alignments/total_peaks*100)}% of temporal peaks align with both frequency anomalies and words",
            "debug_info": {
                "samples_with_transcripts": samples_with_transcripts,
                "samples_with_freq_data": samples_with_freq_data,
                "total_peaks": total_peaks
            }
        })
    else:
        cross_referenced_insights.append({
            "pattern_type": "alignment_statistics",
            "description": "No temporal peaks found for cross-reference analysis",
            "debug_info": {
                "samples_with_transcripts": samples_with_transcripts,
                "samples_with_freq_data": samples_with_freq_data,
                "total_samples_processed": len(all_sample_data)
            }
        })
    
    # Add global summary
    cross_referenced_insights.append({
        "pattern_type": "global_summary",
        "description": f"Analyzed {temporal_patterns['real']['n_samples']} real and {temporal_patterns['fake']['n_samples']} fake samples",
        "top_risk_words": [w["word"] for w in linguistic_patterns["high_risk_words"][:5]],
        "total_peaks_analyzed": int(total_peaks)
    })
    
    # Save
    results = {
        "temporal_patterns": temporal_patterns,
        "frequency_analysis": frequency_analysis,
        "linguistic_patterns": linguistic_patterns,
        "cross_referenced_insights": cross_referenced_insights,
        "summary": {
            "n_real_samples": temporal_patterns["real"]["n_samples"],
            "n_fake_samples": temporal_patterns["fake"]["n_samples"],
            "total_samples": temporal_patterns["real"]["n_samples"] + temporal_patterns["fake"]["n_samples"]
        }
    }
    
    output_file = output_dir / "global_xai_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Real: {results['summary']['n_real_samples']}")
    print(f"  Fake: {results['summary']['n_fake_samples']}")
    print(f"  High-risk words: {len(linguistic_patterns['high_risk_words'])}")
    print(f"  Problematic phonemes: {len(linguistic_patterns.get('phoneme_analysis', []))}")
    
    if linguistic_patterns['high_risk_words']:
        print("\nTop 5 High-Risk Words:")
        for i, w in enumerate(linguistic_patterns['high_risk_words'][:5], 1):
            print(f"  {i}. {w['word']} (score: {w['avg_score_when_present']:.3f}, n={w['occurrences']})")
    
    if linguistic_patterns.get('phoneme_analysis'):
        print("\nTop 5 Problematic Phonemes:")
        for i, p in enumerate(linguistic_patterns['phoneme_analysis'][:5], 1):
            print(f"  {i}. '{p['phoneme']}' - Fake: {p['mispronounced_pct_fake']:.1f}%, Real: {p['mispronounced_pct_real']:.1f}%")


if __name__ == "__main__":
    main()
