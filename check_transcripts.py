"""
Check if test audio files have corresponding transcripts
"""
import pandas as pd
from pathlib import Path
import re

print("=" * 80)
print("CHECKING TRANSCRIPT COVERAGE FOR TEST SETS")
print("=" * 80)

# Load test CSVs
print("\n[1] Loading test CSVs...")
test_real = pd.read_csv("metadata/fs_test_real.labeled.csv")
test_fake = pd.read_csv("metadata/fs_test_fake_mms.labeled.csv")
print(f"✓ Test real: {len(test_real)} samples")
print(f"✓ Test fake: {len(test_fake)} samples")

# Load transcript CSVs
print("\n[2] Loading transcript CSVs...")
fleurs = pd.read_csv("metadata/thirdparty_real_test.fleurs.csv")
from_train = pd.read_csv("metadata/test_real.from_train.roundrobin_least_damage.csv")
master_fake = pd.read_csv("metadata/master_fake.csv")
print(f"✓ FLEURS: {len(fleurs)} samples with transcripts")
print(f"✓ From train: {len(from_train)} samples with transcripts")
print(f"✓ Master fake: {len(master_fake)} samples with transcripts")

# Function to extract utt_id from path
def extract_utt_id(path):
    """Extract utt_id from file path"""
    if pd.isna(path):
        return None
    # Get filename without extension
    filename = Path(path).stem
    return filename

# Check REAL samples
print("\n" + "=" * 80)
print("CHECKING REAL SAMPLES")
print("=" * 80)

# Extract utt_ids from test_real paths
test_real['extracted_utt_id'] = test_real['path_audio'].apply(extract_utt_id)

# Create lookup dictionaries
fleurs_dict = dict(zip(fleurs['utt_id'], fleurs['text']))
from_train_dict = dict(zip(from_train['utt_id'], from_train['text']))

# Check coverage
fleurs_matches = 0
from_train_matches = 0
no_transcript = 0

for idx, row in test_real.iterrows():
    utt_id = row['extracted_utt_id']
    
    if utt_id in fleurs_dict:
        fleurs_matches += 1
    elif utt_id in from_train_dict:
        from_train_matches += 1
    else:
        no_transcript += 1

print(f"\n✓ FLEURS matches: {fleurs_matches} / {len(test_real)} ({fleurs_matches/len(test_real)*100:.1f}%)")
print(f"✓ From train matches: {from_train_matches} / {len(test_real)} ({from_train_matches/len(test_real)*100:.1f}%)")
print(f"✗ No transcript: {no_transcript} / {len(test_real)} ({no_transcript/len(test_real)*100:.1f}%)")
print(f"\n{'='*40}")
print(f"TOTAL COVERAGE: {fleurs_matches + from_train_matches} / {len(test_real)} ({(fleurs_matches + from_train_matches)/len(test_real)*100:.1f}%)")
print(f"{'='*40}")

# Show some examples
print("\n[Examples of matched samples]")
for idx, row in test_real.head(5).iterrows():
    utt_id = row['extracted_utt_id']
    if utt_id in fleurs_dict:
        print(f"  {utt_id}: {fleurs_dict[utt_id][:80]}...")
        break

# Check FAKE samples
print("\n" + "=" * 80)
print("CHECKING FAKE SAMPLES")
print("=" * 80)

# Fake samples already have utt_id column
master_fake_dict = dict(zip(master_fake['utt_id'], master_fake['text']))

fake_matches = 0
fake_no_transcript = 0

for idx, row in test_fake.iterrows():
    utt_id = row['utt_id']
    
    if pd.notna(utt_id) and utt_id in master_fake_dict:
        fake_matches += 1
    else:
        fake_no_transcript += 1

print(f"\n✓ Master fake matches: {fake_matches} / {len(test_fake)} ({fake_matches/len(test_fake)*100:.1f}%)")
print(f"✗ No transcript: {fake_no_transcript} / {len(test_fake)} ({fake_no_transcript/len(test_fake)*100:.1f}%)")
print(f"\n{'='*40}")
print(f"TOTAL COVERAGE: {fake_matches} / {len(test_fake)} ({fake_matches/len(test_fake)*100:.1f}%)")
print(f"{'='*40}")

# Show some examples
print("\n[Examples of matched samples]")
for idx, row in test_fake.head(10).iterrows():
    utt_id = row['utt_id']
    if pd.notna(utt_id) and utt_id in master_fake_dict:
        print(f"  {utt_id}: {master_fake_dict[utt_id][:80]}...")
        break

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
total_test = len(test_real) + len(test_fake)
total_with_transcript = fleurs_matches + from_train_matches + fake_matches
print(f"\nTotal test samples: {total_test}")
print(f"Samples with transcripts: {total_with_transcript} ({total_with_transcript/total_test*100:.1f}%)")
print(f"Samples without transcripts: {total_test - total_with_transcript} ({(total_test - total_with_transcript)/total_test*100:.1f}%)")

print("\nBreakdown:")
print(f"  Real (FLEURS): {fleurs_matches}")
print(f"  Real (from train): {from_train_matches}")
print(f"  Fake: {fake_matches}")
print(f"  Missing: {total_test - total_with_transcript}")

if total_with_transcript / total_test >= 0.95:
    print("\n✅ EXCELLENT: >95% coverage - transcript analysis will be very reliable")
elif total_with_transcript / total_test >= 0.80:
    print("\n✓ GOOD: >80% coverage - transcript analysis will be useful")
elif total_with_transcript / total_test >= 0.50:
    print("\n⚠ MODERATE: >50% coverage - transcript analysis will have limited insights")
else:
    print("\n✗ POOR: <50% coverage - transcript analysis may not be reliable")

print("\n" + "=" * 80)
