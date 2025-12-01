# =======================
# PowerShell Automation
# =======================

$RESULTS_ROOT = "results"
$PY_SCRIPT    = "video_generate_script.py"

# Evaluation settings
$EPISODES     = 5
$VIDEO_LENGTH = 300
$SEED         = 123

# Environments
$ENVS = @(
    "Ant-v5",
    "HalfCheetah-v5",
    "FetchReach-v4"
)

# Algorithms
$ALGOS = @("td3", "sac", "ddpg", "ppo")

Write-Host "Starting video generation..." -ForegroundColor Cyan

foreach ($env_id in $ENVS) {
    foreach ($algo in $ALGOS) {

        Write-Host "`n=== Env: $env_id | Algo: $algo ===" -ForegroundColor Yellow

        # --------------------------------------------------------------------
        # Resolve checkpoint directory in the same way as the Python script:
        #   results/<env_id>/<algo>_<env_id_sanitized>/<latest_run>/checkpoints
        # --------------------------------------------------------------------
        $envDir = Join-Path $RESULTS_ROOT $env_id
        if (-not (Test-Path $envDir)) {
            Write-Host "  [SKIP] Env directory not found: $envDir" -ForegroundColor Red
            continue
        }

        $envSanitized = $env_id.Replace("-", "_").ToLower()
        $algoRootName = "${algo}_${envSanitized}"
        $algoRoot     = Join-Path $envDir $algoRootName

        if (-not (Test-Path $algoRoot)) {
            Write-Host "  [SKIP] Algo directory not found: $algoRoot" -ForegroundColor Red
            continue
        }

        # Get latest run directory (by name) under algoRoot
        $runDirs = Get-ChildItem -Path $algoRoot -Directory | Sort-Object Name
        if ($runDirs.Count -eq 0) {
            Write-Host "  [SKIP] No run directories under: $algoRoot" -ForegroundColor Red
            continue
        }

        $latestRunDir = $runDirs[-1].FullName
        $ckptDir      = Join-Path $latestRunDir "checkpoints"

        if (-not (Test-Path $ckptDir)) {
            Write-Host "  [SKIP] Checkpoints directory not found: $ckptDir" -ForegroundColor Red
            continue
        }

        $bestModelPath = Join-Path $ckptDir "best_model.zip"
        if (-not (Test-Path $bestModelPath)) {
            Write-Host "  [SKIP] best_model.zip not found in: $ckptDir" -ForegroundColor Red
            continue
        }

        Write-Host "  [USE] Checkpoint directory: $ckptDir" -ForegroundColor Green
        Write-Host "  [USE] Model file:          $bestModelPath" -ForegroundColor Green

        # --------------------------------------------------------------------
        # Output video directory
        # --------------------------------------------------------------------
        $videoDir = Join-Path "videos" "$env_id\$algo"
        New-Item -ItemType Directory -Force -Path $videoDir | Out-Null
        Write-Host "  [INFO] Video output dir:   $videoDir"

        # --------------------------------------------------------------------
        # Call python script
        # --------------------------------------------------------------------
        python $PY_SCRIPT `
            --algo $algo `
            --env-id $env_id `
            --checkpoint-dir $ckptDir `
            --video-dir $videoDir `
            --episodes $EPISODES `
            --video-length $VIDEO_LENGTH `
            --seed $SEED

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK]   Video generation completed." -ForegroundColor Green
        }
        else {
            Write-Host "  [FAIL] Python script exited with code $LASTEXITCODE" -ForegroundColor Red
        }
    }
}

Write-Host "`nAll recordings processed (used or skipped as logged above)." -ForegroundColor Cyan
