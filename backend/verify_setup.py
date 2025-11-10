#!/usr/bin/env python3
"""
Setup Verification Script
Run this to verify all components are properly configured.
"""
import sys
from pathlib import Path

def print_status(check_name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {check_name}")
    if message:
        print(f"     └─ {message}")

def main():
    print("=" * 60)
    print("DEEPFAKE DETECTION SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    print()

    all_passed = True

    # Check 1: Checkpoint file exists
    print("📁 FILE CHECKS")
    print("-" * 60)
    ckpt_path = Path("checkpoints/moe_ptm2_v5_aggressive_best.pt")
    passed = ckpt_path.exists()
    all_passed &= passed
    print_status(
        "Checkpoint file exists",
        passed,
        f"Path: {ckpt_path.absolute()}" if passed else "MISSING! Place checkpoint in backend/checkpoints/"
    )

    # Check 2: XAI module exists
    xai_path = Path("xai_analysis.py")
    passed = xai_path.exists()
    all_passed &= passed
    print_status("XAI analysis module", passed)

    # Check 3: Model file updated
    moe_path = Path("moe_model.py")
    if moe_path.exists():
        content = moe_path.read_text()
        passed = "ImprovedExpert" in content and "SE1D" in content
        all_passed &= passed
        print_status("Model architecture updated", passed, "Contains v5 components")
    else:
        print_status("Model file exists", False)
        all_passed = False

    print()

    # Check 4: App.py configuration
    print("⚙️  CONFIGURATION CHECKS")
    print("-" * 60)
    app_path = Path("app.py")
    if app_path.exists():
        content = app_path.read_text()
        
        # Check checkpoint name
        passed = "moe_ptm2_v5_aggressive_best.pt" in content
        all_passed &= passed
        print_status("Checkpoint name updated", passed)
        
        # Check threshold
        passed = "0.111145" in content
        all_passed &= passed
        print_status("Threshold updated", passed)
        
        # Check XAI import
        passed = "from xai_analysis import" in content
        all_passed &= passed
        print_status("XAI module imported", passed)
        
        # Check XAI integration
        passed = "run_complete_xai_analysis" in content
        all_passed &= passed
        print_status("XAI analysis integrated", passed)
    else:
        print_status("app.py exists", False)
        all_passed = False

    print()

    # Check 5: Environment file
    print("🔧 ENVIRONMENT CHECKS")
    print("-" * 60)
    env_path = Path(".env")
    if env_path.exists():
        content = env_path.read_text()
        passed = "0.111145" in content
        all_passed &= passed
        print_status(".env threshold updated", passed)
    else:
        print_status(".env file exists", False, "Create from .env.example if needed")
        all_passed = False

    print()

    # Check 6: Python dependencies
    print("📦 DEPENDENCY CHECKS")
    print("-" * 60)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("librosa", "Librosa"),
        ("transformers", "Transformers"),
        ("faster_whisper", "Faster Whisper")
    ]
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_status(f"{display_name} installed", True)
        except ImportError:
            print_status(f"{display_name} installed", False, "Run: pip install -r requirements.txt")
            all_passed = False

    print()

    # Check 7: Model structure validation
    print("🏗️  MODEL STRUCTURE CHECKS")
    print("-" * 60)
    try:
        from moe_model import MoEModel, ImprovedExpert, SE1D, TinyGate
        print_status("Model classes importable", True)
        
        # Check MoEModel signature
        import inspect
        sig = inspect.signature(MoEModel.__init__)
        params = list(sig.parameters.keys())
        
        required_params = ['ptms', 'in_dim_each', 'expert_bottleneck', 'expert_drop', 
                          'gate_hidden', 'gate_drop', 'use_batchnorm', 'use_se', 
                          'simple_gate', 'stochastic_depth', 'use_fusion', 'fusion_dropout']
        
        has_all = all(p in params for p in required_params)
        print_status("Model signature correct", has_all, "All v5 parameters present")
        all_passed &= has_all
        
    except Exception as e:
        print_status("Model structure", False, str(e))
        all_passed = False

    print()

    # Check 8: XAI functions
    print("🔍 XAI COMPONENT CHECKS")
    print("-" * 60)
    try:
        from xai_analysis import (
            compute_temporal_heatmap,
            compute_frequency_contribution,
            compute_expert_agreement,
            compute_attention_rollout,
            detect_breathing_patterns,
            run_complete_xai_analysis
        )
        print_status("Temporal heatmap function", True)
        print_status("Frequency contribution function", True)
        print_status("Expert agreement function", True)
        print_status("Attention rollout function", True)
        print_status("Breathing patterns function", True)
        print_status("Master XAI function", True)
    except Exception as e:
        print_status("XAI functions", False, str(e))
        all_passed = False

    print()
    print("=" * 60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - System is ready!")
        print()
        print("Next steps:")
        print("  1. Start backend: python -m uvicorn app:app --reload")
        print("  2. Start frontend: cd ../frontend && npm run dev")
        print("  3. Open browser: http://localhost:5173")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
