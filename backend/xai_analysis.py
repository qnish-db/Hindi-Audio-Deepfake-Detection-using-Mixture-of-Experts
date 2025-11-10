# xai_analysis.py
"""
Explainable AI Analysis Module for Deepfake Detection
Handles all visualization computations for frontend rendering.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import librosa
from pathlib import Path
import time


# =========================
# TIER 1: TEMPORAL CONSISTENCY
# =========================
def compute_temporal_heatmap(
    model, 
    wav: np.ndarray, 
    sr: int,
    feature_extractor_fn,
    device: str,
    window_sec: float = 0.5,
    hop_sec: float = 0.25
) -> Dict:
    """
    Slide a window over audio and compute fakeness score for each segment.
    Returns temporal heatmap showing consistency/variation in scores.
    
    Returns:
        {
            'timestamps': List[float],  # Center time of each window
            'scores': List[float],      # Fakeness score per window
            'mean_score': float,
            'std_score': float,
            'consistency_index': float  # Lower = more consistent (suspicious)
        }
    """
    model.eval()
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    timestamps = []
    scores = []
    
    for start in range(0, len(wav) - window_samples + 1, hop_samples):
        end = start + window_samples
        segment = wav[start:end]
        
        # Pad if needed
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)), mode='constant')
        
        # Extract features for this segment
        try:
            feats = feature_extractor_fn(segment)
            xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
            
            with torch.inference_mode():
                logits, _, _ = model(xdict)
                probs = torch.softmax(logits, dim=1)
                score = float(probs[0, 1].item())
            
            timestamps.append((start + end) / 2 / sr)
            scores.append(score)
        except Exception:
            continue
    
    if len(scores) == 0:
        return {
            'timestamps': [],
            'scores': [],
            'mean_score': 0.0,
            'std_score': 0.0,
            'consistency_index': 0.0
        }
    
    scores_arr = np.array(scores)
    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))
    
    # Consistency index: coefficient of variation (lower = more robotic)
    consistency_index = std_score / (mean_score + 1e-8)
    
    return {
        'timestamps': timestamps,
        'scores': scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'consistency_index': consistency_index
    }


# =========================
# TIER 1: FREQUENCY BAND CONTRIBUTION
# =========================
def compute_frequency_contribution(
    wav: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> Dict:
    """
    Analyze frequency distribution and highlight suspicious bands.
    TTS often fails at high frequencies or has unnatural formants.
    
    Returns:
        {
            'mel_spectrogram': List[List[float]],  # (time, freq)
            'freq_bins': List[float],              # Hz values
            'time_bins': List[float],              # Seconds
            'suspicious_bands': List[Dict],         # Bands with anomalies
            'high_freq_energy': float,             # Energy above 8kHz
            'formant_consistency': float           # Formant variation metric
        }
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Frequency bins in Hz
    freq_bins = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr/2)
    time_bins = librosa.frames_to_time(
        np.arange(mel_spec.shape[1]), sr=sr, hop_length=hop_length
    )
    
    # Analyze high frequency energy (above 8kHz)
    high_freq_idx = np.where(freq_bins >= 8000)[0]
    if len(high_freq_idx) > 0:
        high_freq_energy = float(np.mean(mel_spec[high_freq_idx, :]))
    else:
        high_freq_energy = 0.0
    
    # Detect suspicious bands (very uniform energy across time)
    suspicious_bands = []
    for freq_idx in range(n_mels):
        freq_energy = mel_spec[freq_idx, :]
        std_energy = np.std(freq_energy)
        mean_energy = np.mean(freq_energy)
        
        # Low variation = suspicious
        if mean_energy > 0 and (std_energy / (mean_energy + 1e-8)) < 0.2:
            suspicious_bands.append({
                'freq_hz': float(freq_bins[freq_idx]),
                'freq_range': f"{freq_bins[freq_idx-1] if freq_idx > 0 else 0:.0f}-{freq_bins[freq_idx]:.0f} Hz",
                'uniformity': float(std_energy / (mean_energy + 1e-8))
            })
    
    # Formant consistency (simplified)
    formant_consistency = float(np.mean([np.std(mel_spec[i, :]) for i in range(20, 60)]))
    
    return {
        'mel_spectrogram': mel_spec_db.tolist(),
        'freq_bins': freq_bins.tolist(),
        'time_bins': time_bins.tolist(),
        'suspicious_bands': suspicious_bands[:5],  # Top 5
        'high_freq_energy': high_freq_energy,
        'formant_consistency': formant_consistency
    }


# =========================
# TIER 1: EXPERT AGREEMENT
# =========================
def compute_expert_agreement(
    model,
    feats: Dict[str, np.ndarray],
    device: str
) -> Dict:
    """
    Compute individual expert predictions and agreement metrics.
    
    Returns:
        {
            'experts': {
                'hubert-base': {'prob_fake': float, 'prediction': str},
                'wav2vec2-base': {'prob_fake': float, 'prediction': str}
            },
            'agreement_score': float,  # 0-1, higher = more agreement
            'interpretation': str      # Human-readable explanation
        }
    """
    model.eval()
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    
    with torch.inference_mode():
        logits, expert_logits, gate_weights = model(xdict)
    
    # Extract individual expert predictions
    experts = {}
    expert_names = model.ptms
    
    for idx, expert_name in enumerate(expert_names):
        expert_probs = torch.softmax(expert_logits[0, idx, :], dim=0)
        prob_fake = float(expert_probs[1].item())
        
        experts[expert_name] = {
            'prob_fake': prob_fake,
            'prediction': 'FAKE' if prob_fake >= 0.5 else 'REAL',
            'gate_weight': float(gate_weights[0, idx].item())
        }
    
    # Compute agreement score
    probs = [experts[e]['prob_fake'] for e in expert_names]
    agreement_score = 1.0 - np.std(probs)  # Higher std = lower agreement
    
    # Generate interpretation
    all_fake = all(experts[e]['prediction'] == 'FAKE' for e in expert_names)
    all_real = all(experts[e]['prediction'] == 'REAL' for e in expert_names)
    
    if all_fake:
        interpretation = "Both experts detected strong anomalies (acoustic + linguistic)"
    elif all_real:
        interpretation = "Both experts found the audio natural"
    else:
        # Mixed signals
        fake_expert = [e for e in expert_names if experts[e]['prediction'] == 'FAKE'][0]
        if 'hubert' in fake_expert.lower():
            interpretation = "Acoustic anomalies detected, but linguistic patterns appear natural"
        else:
            interpretation = "Linguistic patterns suspicious, but acoustics appear natural"
    
    return {
        'experts': experts,
        'agreement_score': float(agreement_score),
        'interpretation': interpretation
    }


# =========================
# TIER 3: GRADCAM-STYLE ATTENTION
# =========================
def compute_attention_rollout(
    model,
    feats: Dict[str, np.ndarray],
    device: str,
    ptm_models: Dict  # HuBERT and Wav2Vec2 models from ptm_feat
) -> Dict:
    """
    Extract attention weights from PTMs and compute attention rollout.
    Shows which temporal regions the model focused on.
    
    Returns:
        {
            'hubert_attention': List[float],    # Attention per time frame
            'wav2vec2_attention': List[float],
            'combined_attention': List[float],
            'peak_regions': List[Tuple[float, float]]  # (start_sec, end_sec)
        }
    """
    # This is a simplified version - full implementation requires model hooks
    # For now, we'll use a gradient-based approach
    
    model.eval()
    xdict_grad = {k: torch.from_numpy(v)[None, :].to(device).requires_grad_(True) 
                  for k, v in feats.items()}
    
    # Forward pass
    logits, expert_logits, gate_weights = model(xdict_grad)
    
    # Get gradient w.r.t. fake class
    fake_score = logits[0, 1]
    fake_score.backward()
    
    # Compute importance as gradient magnitude
    attention_maps = {}
    for ptm_name, feat_tensor in xdict_grad.items():
        if feat_tensor.grad is not None:
            grad_magnitude = torch.abs(feat_tensor.grad[0]).cpu().numpy()
            # Normalize to [0, 1]
            grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
            attention_maps[ptm_name] = grad_magnitude.tolist()
        else:
            attention_maps[ptm_name] = [0.0] * len(feat_tensor[0])
    
    # Combined attention (weighted by gate weights)
    combined = np.zeros(len(attention_maps[model.ptms[0]]))
    for idx, ptm_name in enumerate(model.ptms):
        weight = float(gate_weights[0, idx].item())
        combined += weight * np.array(attention_maps[ptm_name])
    
    # Find peak regions (top 10% attention)
    threshold = np.percentile(combined, 90)
    peak_indices = np.where(combined >= threshold)[0]
    
    # Group consecutive indices into regions (simplified)
    peak_regions = []
    if len(peak_indices) > 0:
        start_idx = peak_indices[0]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] > 10:  # Gap threshold
                peak_regions.append((int(start_idx), int(peak_indices[i-1])))
                start_idx = peak_indices[i]
        peak_regions.append((int(start_idx), int(peak_indices[-1])))
    
    return {
        'hubert_attention': attention_maps.get('hubert-base', []),
        'wav2vec2_attention': attention_maps.get('wav2vec2-base', []),
        'combined_attention': combined.tolist(),
        'peak_regions': peak_regions
    }


# =========================
# TIER 4: BREATHING PATTERNS
# =========================
def detect_breathing_patterns(
    wav: np.ndarray,
    sr: int,
    top_db: int = 40,
    min_silence_len: float = 0.1
) -> Dict:
    """
    Detect pauses/breathing patterns in speech.
    Real speech: Irregular pauses (0.2-0.8s, varies by phrase)
    TTS: Mechanically regular or absent pauses
    
    Returns:
        {
            'pauses': List[Dict],  # [{start: float, end: float, duration: float}]
            'pause_intervals': List[float],  # Time between pauses
            'regularity_score': float,  # Higher = more regular (suspicious)
            'mean_pause_duration': float,
            'std_pause_duration': float,
            'interpretation': str
        }
    """
    # Detect non-silent intervals
    intervals = librosa.effects.split(
        wav, top_db=top_db, frame_length=2048, hop_length=512
    )
    
    # Convert to pauses (gaps between speech)
    pauses = []
    for i in range(len(intervals) - 1):
        pause_start = intervals[i][1] / sr
        pause_end = intervals[i + 1][0] / sr
        pause_duration = pause_end - pause_start
        
        if pause_duration >= min_silence_len:
            pauses.append({
                'start': float(pause_start),
                'end': float(pause_end),
                'duration': float(pause_duration)
            })
    
    if len(pauses) == 0:
        return {
            'pauses': [],
            'pause_intervals': [],
            'regularity_score': 1.0,  # No pauses = very suspicious
            'mean_pause_duration': 0.0,
            'std_pause_duration': 0.0,
            'interpretation': "No breathing pauses detected - highly suspicious for TTS"
        }
    
    # Compute pause statistics
    durations = [p['duration'] for p in pauses]
    mean_duration = float(np.mean(durations))
    std_duration = float(np.std(durations))
    
    # Compute intervals between pauses
    pause_intervals = []
    for i in range(len(pauses) - 1):
        interval = pauses[i + 1]['start'] - pauses[i]['end']
        pause_intervals.append(float(interval))
    
    # Regularity score: coefficient of variation for intervals
    # Low CV = regular = suspicious
    if len(pause_intervals) > 1:
        interval_std = np.std(pause_intervals)
        interval_mean = np.mean(pause_intervals)
        regularity_score = 1.0 - (interval_std / (interval_mean + 1e-8))
        regularity_score = max(0.0, min(1.0, regularity_score))
    else:
        regularity_score = 0.5
    
    # Interpretation
    if regularity_score > 0.7:
        interpretation = "Breathing patterns are suspiciously regular - typical of TTS"
    elif regularity_score > 0.4:
        interpretation = "Breathing patterns show moderate regularity"
    else:
        interpretation = "Breathing patterns are natural and varied"
    
    return {
        'pauses': pauses,
        'pause_intervals': pause_intervals,
        'regularity_score': float(regularity_score),
        'mean_pause_duration': mean_duration,
        'std_pause_duration': std_duration,
        'interpretation': interpretation
    }


# =========================
# MASTER ANALYSIS FUNCTION
# =========================
def run_complete_xai_analysis(
    model,
    wav: np.ndarray,
    sr: int,
    feats: Dict[str, np.ndarray],
    device: str,
    feature_extractor_fn,
    ptm_models: Optional[Dict] = None
) -> Dict:
    """
    Run all XAI analyses and return comprehensive results.
    
    Args:
        model: MoEModel instance
        wav: Audio waveform (mono, 16kHz)
        sr: Sample rate (16000)
        feats: Pre-extracted PTM features
        device: 'cuda' or 'cpu'
        feature_extractor_fn: Function to extract features from audio segment
        ptm_models: Optional dict of PTM models for attention extraction
    
    Returns:
        {
            'temporal_heatmap': {...},
            'frequency_contribution': {...},
            'expert_agreement': {...},
            'attention_rollout': {...},
            'breathing_patterns': {...},
            'processing_time_ms': float
        }
    """
    start_time = time.perf_counter()
    
    results = {}
    
    try:
        # Tier 1: Temporal Consistency
        results['temporal_heatmap'] = compute_temporal_heatmap(
            model, wav, sr, feature_extractor_fn, device
        )
    except Exception as e:
        results['temporal_heatmap'] = {'error': str(e)}
    
    try:
        # Tier 1: Frequency Contribution
        results['frequency_contribution'] = compute_frequency_contribution(
            wav, sr
        )
    except Exception as e:
        results['frequency_contribution'] = {'error': str(e)}
    
    try:
        # Tier 1: Expert Agreement
        results['expert_agreement'] = compute_expert_agreement(
            model, feats, device
        )
    except Exception as e:
        results['expert_agreement'] = {'error': str(e)}
    
    try:
        # Tier 3: Attention Rollout (GradCAM-style)
        results['attention_rollout'] = compute_attention_rollout(
            model, feats, device, ptm_models or {}
        )
    except Exception as e:
        results['attention_rollout'] = {'error': str(e)}
    
    try:
        # Tier 4: Breathing Patterns
        results['breathing_patterns'] = detect_breathing_patterns(
            wav, sr
        )
    except Exception as e:
        results['breathing_patterns'] = {'error': str(e)}
    
    results['processing_time_ms'] = int((time.perf_counter() - start_time) * 1000)
    
    return results
