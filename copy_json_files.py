"""
Quick script to copy JSON files to frontend
"""
import shutil
from pathlib import Path

# Source and destination
src_dir = Path("global_xai_results")
dst_dir = Path("frontend/public/global_xai_results")

# Ensure destination exists
dst_dir.mkdir(parents=True, exist_ok=True)

# Copy JSON files
json_files = list(src_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files to copy:")

for json_file in json_files:
    dst_file = dst_dir / json_file.name
    shutil.copy2(json_file, dst_file)
    print(f"  ✓ Copied {json_file.name} ({json_file.stat().st_size:,} bytes)")

print(f"\n✅ All files copied to {dst_dir}")
