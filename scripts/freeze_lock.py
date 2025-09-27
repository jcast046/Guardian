import json
from huggingface_hub import HfApi

api = HfApi()
lock = json.load(open("models.lock.json", "r"))
for m in lock["models"]:
    rev = m["revision"]
    info = api.model_info(m["repo_id"], revision=rev)
    m["revision"] = info.sha  # pin to exact commit
json.dump(lock, open("models.lock.json", "w"), indent=2)
print("Pinned revisions updated.")
