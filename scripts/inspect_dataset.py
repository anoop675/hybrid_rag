"""List SciFact dataset files on Hugging Face (notebook section 3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import list_repo_files

for f in sorted(list_repo_files("BeIR/scifact", repo_type="dataset")):
    print(f)
for f in sorted(list_repo_files("BeIR/scifact-qrels", repo_type="dataset")):
    print(f)
