# Repo cleanup, commit and push.
# Run from PowerShell. The Linux sandbox cannot delete on this mount, which is why
# this is a script rather than something already done.
#
#   cd C:\Users\Kedar\projects\divya_maam\lab_repo
#   .\cleanup_and_push.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "`n=== before ===" -ForegroundColor Cyan
"{0:N0} MB" -f ((Get-ChildItem -Recurse -File -Exclude .git | Measure-Object Length -Sum).Sum / 1MB)

# ---------------------------------------------------------------------------
# 1. Exact duplicate trees. Verified byte-identical by md5: every file in these
#    four directories also exists in its parent. full_results_summary.txt was
#    present four times.
# ---------------------------------------------------------------------------
$dupes = @(
  "BTP-Reproduction\results\hpc_results_backup",
  "BTP-Reproduction\results\results",
  "BTP-Reproduction\logs\hpc_logs_backup",
  "BTP-Reproduction\scripts\hpc_scripts_backup",
  "BTP-Reproduction\phase3-mechanism\__pycache__"
)
foreach ($d in $dupes) {
  if (Test-Path $d) { Remove-Item -Recurse -Force $d; Write-Host "removed $d" -ForegroundColor Yellow }
}

# ---------------------------------------------------------------------------
# 2. Per-sample prediction dumps. 117 files, 365 MB, and the aggregate scores
#    they summarise are kept in the *_results.json files beside them.
#    Recoverable from git history if ever needed:
#      git log --all --diff-filter=D --name-only -- "*samples*.jsonl"
#      git checkout <sha>^ -- <path>
# ---------------------------------------------------------------------------
$samples = Get-ChildItem -Recurse -File -Filter "*samples*.jsonl" |
           Where-Object { $_.FullName -notmatch "\\\.git\\" }
if ($samples) {
  $mb = "{0:N0}" -f (($samples | Measure-Object Length -Sum).Sum / 1MB)
  $samples | Remove-Item -Force
  Write-Host "removed $($samples.Count) per-sample dumps, $mb MB" -ForegroundColor Yellow
}

Write-Host "`n=== after ===" -ForegroundColor Cyan
"{0:N0} MB" -f ((Get-ChildItem -Recurse -File -Exclude .git | Measure-Object Length -Sum).Sum / 1MB)

# ---------------------------------------------------------------------------
# 3. Commit
# ---------------------------------------------------------------------------
git add -A

Write-Host "`n=== staged ===" -ForegroundColor Cyan
git status --short | Select-Object -First 25
Write-Host ("... {0} paths staged" -f (git diff --cached --name-only | Measure-Object).Count)

git commit -m @"
Consolidate results, remove duplicated trees, rewrite README

Removed four byte-identical backup trees (results/hpc_results_backup,
results/results, logs/hpc_logs_backup, scripts/hpc_scripts_backup) and the
117 per-sample prediction dumps, which totalled 365 MB and duplicate the
aggregate scores kept alongside them in *_results.json.

Added BTP-Reproduction/RESULTS.md: every result across all three phases in
one place, with conclusions, withdrawn claims and known limitations.

README cut from 381 lines to roughly 150, reorganised around findings,
layout and the gotchas worth knowing.

Phase 3 adds the cause of the Qwen text collapse, the visual token access
threshold measured on two architectures, and a portable tool that measures
it via forward hooks.
"@

# ---------------------------------------------------------------------------
# 4. Push. Credentials live in Windows Credential Manager under the lab account.
#    If this 403s, clear the cached personal credential first:
#      cmdkey /delete:git:https://github.com
# ---------------------------------------------------------------------------
Write-Host "`n=== pushing ===" -ForegroundColor Cyan
git push origin main

git log --oneline -5
