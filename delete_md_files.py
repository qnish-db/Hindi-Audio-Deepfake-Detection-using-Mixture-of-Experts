import os
from pathlib import Path

# List of .md files to delete in the root directory
md_files = [
    'COPY_FILES_INSTRUCTIONS.md',
    'FRONTEND_FIX_NOW.md',
    'GLOBAL_XAI_FINAL_FIX.md',
    'GLOBAL_XAI_FIXES.md',
    'GLOBAL_XAI_FRONTEND_SETUP.md',
    'GLOBAL_XAI_IMPLEMENTATION_PLAN.md',
    'GLOBAL_XAI_QUICK_START.md',
    'LID_FIX_COMPLETE.md',
    'PIPELINE_VERIFICATION.md',
    'PREPROCESSING_FIX_COMPLETE.md',
    'PREPROCESSING_MISMATCH_CRITICAL.md',
    'XAI_FIXES_COMPLETE.md'
]

deleted = 0
for f in md_files:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted: {f}")
        deleted += 1
    else:
        print(f"Not found: {f}")

print(f"\nTotal deleted: {deleted} files")
