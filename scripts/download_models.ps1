param([string]$LockFile = "models.lock.json")
$ErrorActionPreference = "Stop"

pip show huggingface_hub | Out-Null 2>$null; if ($LASTEXITCODE -ne 0) { pip install --upgrade huggingface_hub | Out-Null }

$cfg = Get-Content $LockFile -Raw | ConvertFrom-Json
foreach ($m in $cfg.models) {
  Write-Host ">> $($m.repo_id) @ $($m.revision) → $($m.local_dir)"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${m.repo_id}", revision="${m.revision}", local_dir=r"${m.local_dir}", local_dir_use_symlinks=False)
print("OK")
PY
}
Write-Host "All models downloaded."
