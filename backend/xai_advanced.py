# xai_advanced.py
"""
Advanced XAI Methods: Integrated Gradients, SHAP-like, LRP
These methods provide TIME-BASED attribution (not just feature-based)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time


def compute_integrated_gradients(
    model,
    feats: Dict[str, np.ndarray],
    device: str,
    steps: int = 50
) -> Dict:
    """
    Integrated Gradients: Shows which time segments contributed to FAKE prediction.
    Returns attribution per time frame (not per feature dimension).
    
    Method:
    1. Create baseline (zero vector)
    2. Interpolate from baseline to actual input
    3. Compute gradients at each step
    4. Integrate (sum) gradients
    
    Returns:
        {
            'temporal_attribution': List[float],  # Attribution per time frame
            'attribution_scores': Dict[str, List[float]],  # Per expert
            'peak_timestamps': List[Tuple[float, float]],  # High attribution regions
            'method': 'integrated_gradients'
        }
    """
    model.eval()
    
    # Convert to tensors
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    
    # Get baseline (zeros)
    baseline = {k: torch.zeros_like(v) for k, v in xdict.items()}
    
    # Storage for gradients
    integrated_grads = {k: torch.zeros_like(v) for k, v in xdict.items()}
    
    # Interpolate and accumulate gradients
    for step in range(steps):
        alpha = (step + 1) / steps
        
        # Interpolated input
        interpolated = {
            k: baseline[k] + alpha * (xdict[k] - baseline[k])
            for k in xdict.keys()
        }
        
        # Require gradients
        for v in interpolated.values():
            v.requires_grad_(True)
        
        # Forward pass
        logits, _, _ = model(interpolated)
        fake_score = logits[0, 1]
        
        # Backward
        fake_score.backward()
        
        # Accumulate gradients
        for k in xdict.keys():
            if interpolated[k].grad is not None:
                integrated_grads[k] += interpolated[k].grad
    
    # Average and multiply by (input - baseline)
    for k in xdict.keys():
        integrated_grads[k] = (integrated_grads[k] / steps) * (xdict[k] - baseline[k])
    
    # Compute temporal attribution (sum across feature dimensions for each time frame)
    temporal_attribution = {}
    for ptm_name, grad_tensor in integrated_grads.items():
        # grad_tensor shape: [1, seq_len, feature_dim] or [1, feature_dim]
        grad_np = grad_tensor[0].cpu().detach().numpy()
        
        if len(grad_np.shape) == 2:  # [seq_len, feature_dim]
            # Sum across feature dimension
            time_attr = np.abs(grad_np).sum(axis=1)
        else:  # [feature_dim]
            # Single time point (global pooling already done)
            time_attr = np.abs(grad_np)
        
        # Normalize
        if time_attr.max() > 0:
            time_attr = time_attr / time_attr.max()
        
        temporal_attribution[ptm_name] = time_attr.tolist()
    
    # Combine attributions (simple average)
    ptm_names = list(temporal_attribution.keys())
    if len(ptm_names) > 0:
        # Make all same length
        max_len = max(len(temporal_attribution[p]) for p in ptm_names)
        combined = np.zeros(max_len)
        for ptm_name in ptm_names:
            attr = temporal_attribution[ptm_name]
            # Pad or truncate
            if len(attr) < max_len:
                attr = attr + [0.0] * (max_len - len(attr))
            elif len(attr) > max_len:
                attr = attr[:max_len]
            combined += np.array(attr)
        combined /= len(ptm_names)
    else:
        combined = np.array([])
    
    # Find peak regions (top 20% attribution)
    threshold = np.percentile(combined, 80) if len(combined) > 0 else 0
    peak_indices = np.where(combined >= threshold)[0]
    
    # Group consecutive indices
    peak_regions = []
    if len(peak_indices) > 0:
        start_idx = peak_indices[0]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] > 5:  # Gap threshold
                peak_regions.append((int(start_idx), int(peak_indices[i-1])))
                start_idx = peak_indices[i]
        peak_regions.append((int(start_idx), int(peak_indices[-1])))
    
    return {
        'temporal_attribution': combined.tolist(),
        'attribution_per_expert': temporal_attribution,
        'peak_regions': peak_regions,
        'method': 'integrated_gradients'
    }


def compute_shap_approximation(
    model,
    feats: Dict[str, np.ndarray],
    device: str,
    n_samples: int = 100
) -> Dict:
    """
    SHAP-like approximation: Shows feature importance per expert.
    Uses sampling-based approach (faster than exact SHAP).
    
    Returns temporal importance (which time segments matter most).
    
    Returns:
        {
            'temporal_importance': List[float],  # Importance per time frame
            'expert_contributions': Dict[str, float],  # Overall contribution per expert
            'top_moments': List[Dict],  # Top contributing time segments
            'method': 'shap_approximation'
        }
    """
    model.eval()
    
    # Get baseline prediction (zero input)
    baseline = {k: torch.zeros_like(torch.from_numpy(v)[None, :]).to(device) 
                for k, v in feats.items()}
    with torch.inference_mode():
        baseline_logits, _, _ = model(baseline)
        baseline_pred = torch.softmax(baseline_logits, dim=1)[0, 1].item()
    
    # Get actual prediction
    xdict = {k: torch.from_numpy(v)[None, :].to(device) for k, v in feats.items()}
    with torch.inference_mode():
        actual_logits, expert_logits, gate_weights = model(xdict)
        actual_pred = torch.softmax(actual_logits, dim=1)[0, 1].item()
    
    # Sample-based SHAP approximation
    ptm_names = list(feats.keys())
    contributions = {ptm: 0.0 for ptm in ptm_names}
    
    for _ in range(n_samples):
        # Randomly mask some experts
        mask = np.random.binomial(1, 0.5, len(ptm_names))
        masked_input = {}
        for i, ptm in enumerate(ptm_names):
            if mask[i]:
                masked_input[ptm] = xdict[ptm]
            else:
                masked_input[ptm] = baseline[ptm]
        
        with torch.inference_mode():
            masked_logits, _, _ = model(masked_input)
            masked_pred = torch.softmax(masked_logits, dim=1)[0, 1].item()
        
        # Contribution = change in prediction when this expert is included
        for i, ptm in enumerate(ptm_names):
            if mask[i]:
                contributions[ptm] += (masked_pred - baseline_pred) / n_samples
    
    # Compute temporal importance using gradient
    xdict_grad = {k: torch.from_numpy(v)[None, :].to(device).requires_grad_(True) 
                  for k, v in feats.items()}
    logits, _, _ = model(xdict_grad)
    fake_score = logits[0, 1]
    fake_score.backward()
    
    # Extract temporal importance
    temporal_importance = []
    for ptm_name in ptm_names:
        if xdict_grad[ptm_name].grad is not None:
            grad = torch.abs(xdict_grad[ptm_name].grad[0]).cpu().numpy()
            if len(grad.shape) == 2:  # [seq_len, feat_dim]
                temp_imp = grad.sum(axis=1)
            else:
                temp_imp = grad
            temporal_importance.append(temp_imp)
    
    # Combine temporal importance
    if temporal_importance:
        max_len = max(len(t) for t in temporal_importance)
        combined_temporal = np.zeros(max_len)
        for temp in temporal_importance:
            if len(temp) < max_len:
                temp = np.pad(temp, (0, max_len - len(temp)))
            combined_temporal += temp[:max_len]
        combined_temporal /= len(temporal_importance)
        
        # Normalize
        if combined_temporal.max() > 0:
            combined_temporal = combined_temporal / combined_temporal.max()
    else:
        combined_temporal = np.array([])
    
    # Find top moments (top 10%)
    threshold = np.percentile(combined_temporal, 90) if len(combined_temporal) > 0 else 0
    top_indices = np.where(combined_temporal >= threshold)[0]
    
    top_moments = []
    for idx in top_indices[:10]:  # Top 10
        top_moments.append({
            'frame_index': int(idx),
            'importance': float(combined_temporal[idx])
        })
    
    return {
        'temporal_importance': combined_temporal.tolist(),
        'expert_contributions': contributions,
        'top_moments': top_moments,
        'method': 'shap_approximation'
    }


def compute_lrp_simple(
    model,
    feats: Dict[str, np.ndarray],
    device: str
) -> Dict:
    """
    Simplified Layer-wise Relevance Propagation.
    Backpropagates prediction to show input contribution.
    
    Returns temporal relevance map.
    
    Returns:
        {
            'temporal_relevance': List[float],  # Relevance per time frame
            'relevance_per_expert': Dict[str, List[float]],
            'high_relevance_regions': List[Tuple[int, int]],
            'method': 'lrp_simple'
        }
    """
    model.eval()
    
    # Forward pass with gradient
    xdict = {k: torch.from_numpy(v)[None, :].to(device).requires_grad_(True) 
             for k, v in feats.items()}
    
    logits, expert_logits, gate_weights = model(xdict)
    fake_score = logits[0, 1]
    
    # Backward pass
    fake_score.backward()
    
    # Extract relevance (gradient * input)
    relevance_per_expert = {}
    for ptm_name, feat_tensor in xdict.items():
        if feat_tensor.grad is not None:
            # LRP-0 rule: relevance = gradient * input
            relevance = (feat_tensor.grad * feat_tensor)[0].cpu().detach().numpy()
            
            # Sum across feature dimension to get temporal relevance
            if len(relevance.shape) == 2:  # [seq_len, feat_dim]
                temporal_rel = np.abs(relevance).sum(axis=1)
            else:  # [feat_dim]
                temporal_rel = np.abs(relevance)
            
            # Normalize
            if temporal_rel.max() > 0:
                temporal_rel = temporal_rel / temporal_rel.max()
            
            relevance_per_expert[ptm_name] = temporal_rel.tolist()
    
    # Combine relevance from all experts
    if relevance_per_expert:
        max_len = max(len(relevance_per_expert[p]) for p in relevance_per_expert)
        combined_relevance = np.zeros(max_len)
        for ptm_name, rel in relevance_per_expert.items():
            if len(rel) < max_len:
                rel = rel + [0.0] * (max_len - len(rel))
            combined_relevance += np.array(rel[:max_len])
        combined_relevance /= len(relevance_per_expert)
    else:
        combined_relevance = np.array([])
    
    # Find high relevance regions (top 15%)
    threshold = np.percentile(combined_relevance, 85) if len(combined_relevance) > 0 else 0
    high_indices = np.where(combined_relevance >= threshold)[0]
    
    high_regions = []
    if len(high_indices) > 0:
        start = high_indices[0]
        for i in range(1, len(high_indices)):
            if high_indices[i] - high_indices[i-1] > 10:
                high_regions.append((int(start), int(high_indices[i-1])))
                start = high_indices[i]
        high_regions.append((int(start), int(high_indices[-1])))
    
    return {
        'temporal_relevance': combined_relevance.tolist(),
        'relevance_per_expert': relevance_per_expert,
        'high_relevance_regions': high_regions,
        'method': 'lrp_simple'
    }


def run_advanced_xai(
    model,
    feats: Dict[str, np.ndarray],
    device: str
) -> Dict:
    """
    Run all advanced XAI methods.
    These methods reuse already-extracted features (no re-extraction).
    
    Returns:
        {
            'integrated_gradients': {...},
            'shap_approximation': {...},
            'lrp': {...},
            'processing_time_ms': int
        }
    """
    start_time = time.perf_counter()
    
    results = {}
    
    try:
        results['integrated_gradients'] = compute_integrated_gradients(
            model, feats, device, steps=30  # Reduced for speed
        )
    except Exception as e:
        results['integrated_gradients'] = {'error': str(e)}
    
    try:
        results['shap_approximation'] = compute_shap_approximation(
            model, feats, device, n_samples=50  # Reduced for speed
        )
    except Exception as e:
        results['shap_approximation'] = {'error': str(e)}
    
    try:
        results['lrp'] = compute_lrp_simple(model, feats, device)
    except Exception as e:
        results['lrp'] = {'error': str(e)}
    
    results['processing_time_ms'] = int((time.perf_counter() - start_time) * 1000)
    
    return results
