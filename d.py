"""
PRE-RUN CHECK for global_xai_CORRECT.py
Validates all required files exist before running expensive analysis
"""
import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("PRE-RUN CHECK: Validating files for global_xai_CORRECT.py")
print("=" * 80)

# Paths
CHECKPOINT = "checkpoints/moe_ptm2_v5_aggressive_best.pt"
TEST_REAL_CSV = "metadata/fs_test_real.labeled.csv"
TEST_FAKE_CSV = "metadata/fs_test_fake_mms.labeled.csv"
MASTER_REAL_FLEURS = "metadata/thirdparty_real_test.fleurs.csv"
MASTER_REAL_TRAIN = "metadata/test_real.from_train.roundrobin_least_damage.csv"
MASTER_FAKE = "metadata/master_fake.csv"

errors = []
warnings = []

# Check 1: Checkpoint exists
print("\n[1/7] Checking checkpoint...")
if not Path(CHECKPOINT).exists():
    errors.append(f"❌ Checkpoint not found: {CHECKPOINT}")
else:
    print(f"✓ Checkpoint exists: {CHECKPOINT}")

# Check 2: Test CSVs exist
print("\n[2/7] Checking test CSVs...")
for csv_path in [TEST_REAL_CSV, TEST_FAKE_CSV]:
    if not Path(csv_path).exists():
        errors.append(f"❌ CSV not found: {csv_path}")
    else:
        print(f"✓ CSV exists: {csv_path}")

# Check 3: Master CSVs exist
print("\n[3/7] Checking master CSVs (for transcripts)...")
for csv_path in [MASTER_REAL_FLEURS, MASTER_REAL_TRAIN, MASTER_FAKE]:
    if not Path(csv_path).exists():
        errors.append(f"❌ CSV not found: {csv_path}")
    else:
        print(f"✓ CSV exists: {csv_path}")

# Check 4: PTM feature files exist
print("\n[4/7] Checking PTM feature files (sample 10 from each)...")
test_real = pd.read_csv(TEST_REAL_CSV)
test_fake = pd.read_csv(TEST_FAKE_CSV)

missing_features = 0
for label, df in [("real", test_real.head(10)), ("fake", test_fake.head(10))]:
    for idx, row in df.iterrows():
        for col in ['vec_wav2vec2-base', 'vec_hubert-base']:
            if col in row:
                vec_path = row[col]
                if pd.notna(vec_path):
                    # Apply path resolution
                    if "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm" in vec_path:
                        vec_path = vec_path.replace(
                            "G:\\My Drive\\hindi_dfake\\processed\\features\\ptm",
                            r"C:\Users\pc 1\hindi_df\ptm"
                        )
                    
                    if not Path(vec_path).exists():
                        missing_features += 1
                        if missing_features <= 3:  # Show first 3
                            warnings.append(f"⚠ Feature not found: {vec_path}")

if missing_features > 0:
    errors.append(f"❌ {missing_features} PTM feature files missing!")
else:
    print(f"✓ All sampled PTM features exist")

# Check 5: Audio files - DO WE NEED THEM?
print("\n[5/7] Checking if audio files are needed...")
print("⚠ IMPORTANT: Script loads audio for frequency_contribution()")
print("   But perturbation approximation only needs FEATURES (no audio)")
print("   → Audio is OPTIONAL if you skip frequency analysis")

# Audio path resolution (from runner scripts)
def resolve_audio_path(path_str):
    """Resolve audio path - they're in G drive processed/wav/strong/"""
    if pd.isna(path_str):
        return None
    
    # Already correct path
    if Path(path_str).exists():
        return path_str
    
    # Path is in G drive - check if accessible
    if "G:\\My Drive\\hindi_dfake" in path_str or "G:/My Drive/hindi_dfake" in path_str:
        return path_str if Path(path_str).exists() else None
    
    # Try to construct G drive path
    if "/processed/wav/strong/" in path_str:
        tail = path_str.split("/processed/wav/strong/")[1]
        g_path = f"G:\\My Drive\\hindi_dfake\\processed\\wav\\strong\\{tail}"
        return g_path if Path(g_path).exists() else None
    
    return None

audio_missing = 0
audio_on_gdrive = 0
audio_accessible = 0

for label, df in [("real", test_real.head(10)), ("fake", test_fake.head(10))]:
    for idx, row in df.iterrows():
        audio_path = row.get('path_audio')
        resolved = resolve_audio_path(audio_path)
        
        if resolved is None:
            audio_missing += 1
        elif "G:\\My Drive" in resolved or "G:/My Drive" in resolved:
            audio_on_gdrive += 1
            if Path(resolved).exists():
                audio_accessible += 1
        else:
            if Path(resolved).exists():
                audio_accessible += 1

print(f"  Audio files: {audio_accessible} accessible, {audio_on_gdrive} on G drive, {audio_missing} missing")

if audio_accessible == 0 and audio_on_gdrive > 0:
    warnings.append(f"⚠ All audio on G drive - check if G drive is mounted")
elif audio_missing > 0:
    warnings.append(f"⚠ {audio_missing}/20 sampled audio files missing")
    warnings.append("   → Frequency analysis will be skipped for missing files")
else:
    print(f"✓ All sampled audio files accessible")

# Check 6: Transcripts coverage
print("\n[6/7] Checking transcript coverage...")
master_real_fleurs = pd.read_csv(MASTER_REAL_FLEURS)
master_real_train = pd.read_csv(MASTER_REAL_TRAIN)
master_fake = pd.read_csv(MASTER_FAKE)

transcript_lookup = {}
for _, row in master_real_fleurs.iterrows():
    transcript_lookup[row['utt_id']] = row['text']
for _, row in master_real_train.iterrows():
    transcript_lookup[row['utt_id']] = row['text']
for _, row in master_fake.iterrows():
    transcript_lookup[row['utt_id']] = row['text']

transcripts_found = 0
for label, df in [("real", test_real.head(10)), ("fake", test_fake.head(10))]:
    for idx, row in df.iterrows():
        audio_path = row.get('path_audio')
        if pd.notna(audio_path):
            utt_id = Path(audio_path).stem
            if utt_id in transcript_lookup:
                transcripts_found += 1

print(f"✓ Transcripts found: {transcripts_found}/20 samples")
if transcripts_found < 15:
    warnings.append(f"⚠ Low transcript coverage: {transcripts_found}/20")

# Check 7: Dependencies
print("\n[7/7] Checking Python dependencies...")
try:
    import torch
    print(f"✓ torch: {torch.__version__}")
except:
    errors.append("❌ torch not installed")

try:
    import librosa
    print(f"✓ librosa: {librosa.__version__}")
except:
    warnings.append("⚠ librosa not installed (needed for frequency analysis)")

try:
    from scipy import stats
    print(f"✓ scipy installed")
except:
    errors.append("❌ scipy not installed")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if errors:
    print(f"\n❌ CRITICAL ERRORS ({len(errors)}):")
    for err in errors:
        print(f"  {err}")
    print("\n⛔ DO NOT RUN - Fix errors first!")
else:
    print("\n✅ All critical checks passed!")

if warnings:
    print(f"\n⚠ WARNINGS ({len(warnings)}):")
    for warn in warnings:
        print(f"  {warn}")
    print("\n⚠ Script may run with limited functionality")

if not errors:
    print("\n" + "=" * 80)
    print("READY TO RUN:")
    print("  python global_xai_CORRECT.py --checkpoint checkpoints/moe_ptm2_v5_aggressive_best.pt --max-samples 10")
    print("=" * 80)

print()
