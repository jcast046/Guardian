#!/usr/bin/env python3
"""Model revision locking utility for Guardian project.

This script updates the models.lock.json file to pin model revisions to exact
commit hashes instead of branch names or tags. This ensures reproducible
model downloads and prevents issues with moving references.

Usage:
    python scripts/freeze_lock.py

Author: Joshua Castillo
"""

import json
from huggingface_hub import HfApi

def freeze_model_revisions():
    """Pin model revisions to exact commit hashes for reproducibility.
    
    Reads models.lock.json and updates each model's revision to the exact
    commit SHA instead of branch names or tags. This ensures reproducible
    model downloads.
    
    Raises:
        FileNotFoundError: If models.lock.json doesn't exist
        json.JSONDecodeError: If models.lock.json contains invalid JSON
        Exception: If model info cannot be retrieved from Hugging Face Hub
    """
    api = HfApi()
    
    # Load current lock file
    with open("models.lock.json", "r") as f:
        lock = json.load(f)
    
    # Update each model's revision to exact commit SHA
    for model in lock["models"]:
        current_rev = model["revision"]
        info = api.model_info(model["repo_id"], revision=current_rev)
        model["revision"] = info.sha  # Pin to exact commit
    
    # Write updated lock file
    with open("models.lock.json", "w") as f:
        json.dump(lock, f, indent=2)
    
    print("Pinned revisions updated.")

if __name__ == "__main__":
    freeze_model_revisions()
