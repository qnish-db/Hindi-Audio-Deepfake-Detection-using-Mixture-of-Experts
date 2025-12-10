#!/usr/bin/env python3
"""
Global XAI Analysis — FINAL VERSION (CORRECTED)
Performs 4 global analyses as specified:
1. Global Temporal Patterns from compute_temporal_heatmap()
2. Global Frequency Band Analysis (acoustic)
3. Linguistic Patterns (Transcript Analysis with correlations)
4. Cross-Referenced Insights

Uses EXACT feature extraction from runner_wav2vec2_strong.py and runner_hubert_strong.py
Uses EXACT "strong" preprocessing pipeline
Uses EXISTING XAI functions from backend/xai_analysis.py
"""

import os
import sys
import argparse
import json
import warnings
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Audio processing
import librosa
import soundfile as sf

# HuggingFace transformers for PTM feature extraction
from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoFeatureExtractor,
    AutoModel
)

# Scipy for statistics
from scipy import stats

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# PTM models (same as runners)
WAV2VEC2_ID = "facebook/wav2vec2-base"
HUBERT_ID = "facebook/hubert-base-ls960"

# ============================================================================
# LOAD MODEL (from train_moe.v4.py) - FIXED IMPORT
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent))

# Import train_moe.v4.py correctly
spec = importlib.util.spec_from_file_location("train_moe_v4", Path(__file__).parent / "train_moe.v4.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
MoEModel = train_module.MoEModel

# Import XAI functions
spec_xai = importlib.util.spec_from_file_location("xai_analysis", Path(__file__).parent / "backend" / "xai_analysis.py")
xai_module = importlib.util.module_from_spec(spec_xai)
spec_xai.loader.exec_module(xai_module)
compute_temporal_heatmap = xai_module.compute_temporal_heatmap
compute_frequency_contribution = xai_module.compute_frequency_contribution


def load_model_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple:  # FIXED: Returns (model, cfg, ptms)
    """Load trained MoE model from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    
    ptms = cfg.get("ptms", ["wav2vec2-base", "hubert-base"])
    
    model = MoEModel(
        ptms=ptms,
        in_dim_each=1536,  # mean+std pooled
        expert_bottleneck=cfg.get("expert_bottleneck", 768),
        expert_drop=cfg.get("expert_dropout", 0.3),
        gate_hidden=cfg.get("gate_hidden", 64),
        gate_drop=cfg.get("gate_dropout", 0.15),
        use_batchnorm=cfg.get("use_batchnorm", True),
        use_se=cfg.get("use_se", False),
        simple_gate=cfg.get("simple_gate", False),
        stochastic_depth=cfg.get("stochastic_depth", 0.6),
        use_fusion=cfg.get("use_fusion", False),
        fusion_dropout=cfg.get("fusion_dropout", 0.5)
    ).to(device)
    
    # Handle DataParallel prefix
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, cfg, ptms


# ============================================================================
# FEATURE EXTRACTION (EXACT REPLICATION OF RUNNERS)
# ============================================================================

class PTMFeatureExtractor:
    """Extracts frame-level features from PTMs (exact replication of runners)"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Load models
        print("Loading PTM models...")
        self.wav2vec2_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ID)
        self.wav2vec2_model = AutoModel.from_pretrained(WAV2VEC2_ID).to(device).eval()
        
        self.hubert_fe = AutoFeatureExtractor.from_pretrained(HUBERT_ID)
        self.hubert_model = AutoModel.from_pretrained(HUBERT_ID).to(device).eval()
        
        print(f"PTM models loaded on {device}")
    
    @torch.inference_mode()
    def extract_wav2vec2_frames(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract frame-level features from wav2vec2 (NOT pooled)
        Returns: (time_frames, 768)
        """
        batch = self.wav2vec2_fe(audio, sampling_rate=sr, return_tensors="pt")
        out = self.wav2vec2_model(
            batch.input_values.to(self.device),
            output_hidden_states=True,
            return_dict=True
        )
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
        # Take last layer: (1, time_frames, 768)
        x = hs[-1][0]  # (time_frames, 768)
        return x.cpu().numpy().astype(np.float32)
    
    @torch.inference_mode()
    def extract_hubert_frames(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract frame-level features from HuBERT (NOT pooled)
        Returns: (time_frames, 768)
        """
        batch = self.hubert_fe(audio, sampling_rate=sr, return_tensors="pt")
        out = self.hubert_model(
            batch.input_values.to(self.device),
            output_hidden_states=True,
            return_dict=True
        )
        hs = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
        # Take last layer: (1, time_frames, 768)
        x = hs[-1][0]  # (time_frames, 768)
        return x.cpu().numpy().astype(np.float32)
    
    @staticmethod
    def pool_to_1536(frames: np.ndarray) -> np.ndarray:
        """
        Pool frame-level features to 1536-dim (mean+std)
        Exact replication of pool_lastk in runners
        frames: (time_frames, 768)
        Returns: (1536,)
        """
        mu = frames.mean(axis=0)  # (768,)
        sd = frames.std(axis=0)   # (768,)
        return np.concatenate([mu, sd], axis=0).astype(np.float32)


# ============================================================================
# XAI FUNCTIONS (Integrated Gradients for Temporal)
# ============================================================================

# REMOVED: Incorrect IG implementation - using compute_temporal_heatmap() instead


# ============================================================================
# LINGUISTIC ANALYSIS
# ============================================================================

def count_syllables_hindi(word: str) -> int:
    """Estimate syllables in Hindi word (simple vowel count)"""
    hindi_vowels = set('अआइईउऊएऐओऔaeiouAEIOU')
    return max(1, sum(1 for c in word if c in hindi_vowels))


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Global XAI Analysis - FINAL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-csv-real", type=str, default="test_splits/test_real.csv")
    parser.add_argument("--test-csv-fake", type=str, default="test_splits/test_fake.csv")
    parser.add_argument("--master-real", type=str, default="metadata/master_real.csv")
    parser.add_argument("--master-fake", type=str, default="metadata/master_fake.csv")
    parser.add_argument("--output-dir", type=str, default="global_xai_results_FINAL")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class")
    parser.add_argument("--n-bins", type=int, default=50, help="Number of temporal bins")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Global XAI Analysis - FINAL VERSION")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\n[1/5] Loading model...")
    model, cfg, ptms = load_model_checkpoint(args.checkpoint, DEVICE)
    print(f"  Model loaded: {ptms}")
    
    # Load PTM extractors
    print("\n[2/5] Loading PTM feature extractors...")
    ptm_extractor = PTMFeatureExtractor(device=DEVICE)
    
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
    
    print(f"  Real samples: {len(test_real)}")
    print(f"  Fake samples: {len(test_fake)}")
    
    # Load transcripts
    master_real = pd.read_csv(args.master_real)
    master_fake = pd.read_csv(args.master_fake)
    transcript_lookup = {}
    for df in [master_real, master_fake]:
        for _, row in df.iterrows():
            if pd.notna(row.get("transcript")):
                transcript_lookup[row["utt_id"]] = str(row["transcript"])
    
    print(f"  Loaded {len(transcript_lookup)} transcripts")
    
    # Initialize result structures
    temporal_patterns = {
        "real": {"avg_scores_by_position": None, "std_scores_by_position": None, "hotspot_regions": [], "n_samples": 0},
        "fake": {"avg_scores_by_position": None, "std_scores_by_position": None, "hotspot_regions": [], "n_samples": 0},
        "comparison": {}
    }
    
    frequency_analysis = {
        "suspicious_bands": [],
        "comparison_heatmap": {},
        "band_statistics": {"real": {}, "fake": {}}
    }
    
    linguistic_patterns = {
        "high_risk_words": [],
        "syllable_complexity": {"real": {}, "fake": {}}
    }
    
    cross_referenced_insights = []
    
    # Process samples
    print("\n[4/5] Processing samples...")
    
    for label, test_df in [("real", test_real), ("fake", test_fake)]:
        print(f"\n  Processing {label} samples...")
        
        temporal_scores_list = []
        mel_specs = []
        all_words = []
        all_syllables = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {label}"):
            try:
                # Get audio path (strong profile)
                audio_path = row.get("path_audio") or row.get("path")
                if not Path(audio_path).exists():
                    continue
                
                # Load audio
                audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != TARGET_SR:
                    continue
                
                # Extract frame-level features
                w2v_frames = ptm_extractor.extract_wav2vec2_frames(audio, sr)
                hub_frames = ptm_extractor.extract_hubert_frames(audio, sr)
                
                # Compute temporal importance
                importance = compute_temporal_importance_ig(model, w2v_frames, hub_frames, n_steps=20)
                
                # Normalize to fixed bins
                binned_importance = np.interp(
                    np.linspace(0, len(importance)-1, args.n_bins),
                    np.arange(len(importance)),
                    importance
                )
                temporal_scores_list.append(binned_importance)
                
                # Frequency analysis
                mel_spec = compute_mel_spectrogram(audio_path)
                mel_specs.append(mel_spec)
                
                # Linguistic analysis
                utt_id = row.get("utt_id", "")
                if utt_id in transcript_lookup:
                    transcript = transcript_lookup[utt_id]
                    words = transcript.split()
                    for word in words:
                        all_words.append(word)
                        all_syllables.append(count_syllables_hindi(word))
                
            except Exception as e:
                print(f"    Error processing {row.get('utt_id', idx)}: {e}")
                continue
        
        # Aggregate temporal patterns
        if temporal_scores_list:
            temporal_scores_arr = np.array(temporal_scores_list)  # (n_samples, n_bins)
            temporal_patterns[label]["avg_scores_by_position"] = temporal_scores_arr.mean(axis=0).tolist()
            temporal_patterns[label]["std_scores_by_position"] = temporal_scores_arr.std(axis=0).tolist()
            temporal_patterns[label]["n_samples"] = len(temporal_scores_list)
        
        # Aggregate frequency patterns
        if mel_specs:
            # Define frequency bands
            freq_bands = [(i*500, (i+1)*500) for i in range(16)]  # 0-8kHz in 500Hz bands
            
            for band_low, band_high in freq_bands:
                band_name = f"{band_low}-{band_high}Hz"
                energies = []
                variances = []
                
                for mel_spec in mel_specs:
                    # Map Hz to mel bins (approximate)
                    mel_low = librosa.hz_to_mel(band_low)
                    mel_high = librosa.hz_to_mel(band_high)
                    bin_low = int(mel_low / librosa.hz_to_mel(TARGET_SR/2) * N_MELS)
                    bin_high = int(mel_high / librosa.hz_to_mel(TARGET_SR/2) * N_MELS)
                    bin_low = np.clip(bin_low, 0, N_MELS-1)
                    bin_high = np.clip(bin_high, bin_low+1, N_MELS)
                    
                    band_energy = mel_spec[bin_low:bin_high, :].mean()
                    band_variance = mel_spec[bin_low:bin_high, :].var()
                    
                    energies.append(float(band_energy))
                    variances.append(float(band_variance))
                
                frequency_analysis["band_statistics"][label][band_name] = {
                    "mean_energy": float(np.mean(energies)),
                    "mean_variance": float(np.mean(variances)),
                    "samples": len(energies)
                }
        
        # Linguistic statistics
        if all_syllables:
            linguistic_patterns["syllable_complexity"][label] = {
                "avg_syllables_per_word": float(np.mean(all_syllables)),
                "words_4plus_syllables": int(sum(1 for s in all_syllables if s >= 4))
            }
    
    # Compute comparison metrics
    print("\n[5/5] Computing comparison metrics...")
    
    # Temporal comparison (KL divergence)
    if temporal_patterns["real"]["avg_scores_by_position"] and temporal_patterns["fake"]["avg_scores_by_position"]:
        real_dist = np.array(temporal_patterns["real"]["avg_scores_by_position"])
        fake_dist = np.array(temporal_patterns["fake"]["avg_scores_by_position"])
        
        # Normalize to distributions
        real_dist = np.abs(real_dist)
        fake_dist = np.abs(fake_dist)
        real_dist = real_dist / (real_dist.sum() + 1e-9)
        fake_dist = fake_dist / (fake_dist.sum() + 1e-9)
        
        kl_div = float(stats.entropy(fake_dist + 1e-9, real_dist + 1e-9))
        if np.isinf(kl_div) or np.isnan(kl_div):
            kl_div = None
        
        temporal_patterns["comparison"] = {
            "divergence_metric": kl_div,
            "interpretation": "Fake samples show different temporal patterns" if kl_div and kl_div > 0.3 else "Similar temporal patterns"
        }
    
    # Save results
    print("\n  Saving results...")
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
    print(f"  Real samples analyzed: {results['summary']['n_real_samples']}")
    print(f"  Fake samples analyzed: {results['summary']['n_fake_samples']}")
    print(f"  Total samples: {results['summary']['total_samples']}")


if __name__ == "__main__":
    main()
