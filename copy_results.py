import shutil
import os
from pathlib import Path

# Copy global XAI results to frontend with new name
src = r"global_xai_results_FINAL\global_xai_results.json"
dst = r"frontend\public\global_xai_results_v2.json"

shutil.copy(src, dst)
print(f"✅ Copied {src} to {dst}")

src = Path("global_xai_results_FINAL")
dst = Path("frontend/public/global_xai_results_FINAL")

# Create destination if it doesn't exist
dst.mkdir(parents=True, exist_ok=True)

# Copy all files
for file in src.glob("*"):
    if file.is_file():
        shutil.copy2(file, dst / file.name)
        print(f"Copied: {file.name}")

print("\n✅ Results copied to frontend/public/")
