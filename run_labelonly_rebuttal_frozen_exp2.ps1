
#smoke test
# param(
#     [string]$PythonPath = " "
# )

# $ErrorActionPreference = "Stop"

# $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Set-Location $ScriptDir

# Write-Host "====================================================="
# Write-Host "CivilComments frozen-cache smoke test"
# Write-Host "Working directory: $ScriptDir"
# Write-Host "Python: $PythonPath"
# Write-Host "====================================================="

# & $PythonPath `
#   ".\build_civilcomments_frozen_cache.py" `
#   --data_root ".\data_text_smoke" `
#   --dataset_id "google/civil_comments" `
#   --train_split "train" `
#   --test_split "test" `
#   --model_name "sentence-transformers/all-mpnet-base-v2" `
#   --backbone_tag "all_mpnet_base_v2" `
#   --n_train 2000 `
#   --n_test 1000 `
#   --toxicity_threshold 0.5 `
#   --identity_threshold 0.5 `
#   --min_cell_count 5 `
#   --batch_size 128 `
#   --max_length 128 `
#   --device "auto" `
#   --seed 2026 `
#   --compress 0

# if ($LASTEXITCODE -ne 0) {
#     throw "CivilComments smoke test failed with exit code $LASTEXITCODE"
# }

# Write-Host "====================================================="
# Write-Host "Smoke test completed successfully."
# Write-Host "Caches are under:"
# Write-Host "$ScriptDir\data_text_smoke\feature_cache"
# Write-Host "====================================================="

# powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_labelonly_rebuttal_frozen.ps1"



#——————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————
# param(
#     [string]$PythonPath = " "
# )

# $ErrorActionPreference = "Stop"

# $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Set-Location $ScriptDir

# Write-Host "====================================================="
# Write-Host "Building CivilComments frozen-language cache"
# Write-Host "Working directory: $ScriptDir"
# Write-Host "Python: $PythonPath"
# Write-Host "====================================================="

# & $PythonPath `
#   ".\build_civilcomments_frozen_cache.py" `
#   --data_root ".\data" `
#   --dataset_id "google/civil_comments" `
#   --train_split "train" `
#   --test_split "test" `
#   --model_name "sentence-transformers/all-mpnet-base-v2" `
#   --backbone_tag "all_mpnet_base_v2" `
#   --n_train 40000 `
#   --n_test 10000 `
#   --toxicity_threshold 0.5 `
#   --identity_threshold 0.5 `
#   --min_cell_count 100 `
#   --batch_size 128 `
#   --max_length 128 `
#   --device "auto" `
#   --seed 2026 `
#   --compress 0

# if ($LASTEXITCODE -ne 0) {
#     throw "CivilComments cache construction failed with exit code $LASTEXITCODE"
# }

# Write-Host "====================================================="
# Write-Host "CivilComments frozen cache completed."
# Write-Host "Expected files:"
# Write-Host "$ScriptDir\data\feature_cache\civilcomments_train_all_mpnet_base_v2_pt1.npz"
# Write-Host "$ScriptDir\data\feature_cache\civilcomments_test_all_mpnet_base_v2_pt1.npz"
# Write-Host "$ScriptDir\data\feature_cache\civilcomments_all_mpnet_base_v2_metadata.json"
# Write-Host "====================================================="



# 
# powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_labelonly_rebuttal_frozen.ps1"
# 


#——————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————

param(
    [string]$PythonPath = " "
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$TrainCache = ""
$TestCache = ""

if (-not (Test-Path $TrainCache)) {
    throw "Missing train cache: $TrainCache"
}

if (-not (Test-Path $TestCache)) {
    throw "Missing test cache: $TestCache"
}

Write-Host "====================================================="
Write-Host "Running CivilComments label-only core experiments"
Write-Host "Fixed official split; 10 algorithmic fitting seeds"
Write-Host "====================================================="

& $PythonPath `
  ".\run_labelonly_rebuttal_experiments.py" `
  --data_root ".\data" `
  --datasets "civilcomments" `
  --backbone "all_mpnet_base_v2" `
  --pretrained 1 `
  --max_train 0 `
  --max_test 0 `
  --fast 0 `
  --q_grid "1,2,4,8,16,32" `
  --q_eval 8 `
  --probes "linear,quadratic" `
  --acc_tols "0.005,0.01" `
  --nll_tols "0.01,0.02" `
  --do_stage1_equivalence 1 `
  --do_safe_overlap 1 `
  --do_quotient_leakage 1 `
  --do_frontier_budget 1 `
  --do_alt_probes 0 `
  --do_sensitivity 0 `
  --num_avg 8 `
  --sensitivity_num_avg 4 `
  --seed_list "42,123,3407,2024,2025,7,17,29,314,2718" `
  --outdir ".\runs_labelonly_rebuttal_text_core"

if ($LASTEXITCODE -ne 0) {
    throw "CivilComments core experiments failed with exit code $LASTEXITCODE"
}

Write-Host "====================================================="
Write-Host "CivilComments core experiments completed."
Write-Host "Outputs:"
Write-Host "$ScriptDir\runs_labelonly_rebuttal_text_core"
Write-Host "====================================================="







#——————————————————————————————————————————————————————————
#——————————————————————————————————————————————————————————
# param(
#     [string]$PythonPath = " "
# )

# $ErrorActionPreference = "Stop"

# $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Set-Location $ScriptDir

# $TrainCache = ".\data\feature_cache\civilcomments_train_all_mpnet_base_v2_pt1.npz"
# $TestCache = ".\data\feature_cache\civilcomments_test_all_mpnet_base_v2_pt1.npz"

# if (-not (Test-Path $TrainCache)) {
#     throw "Missing train cache: $TrainCache"
# }

# if (-not (Test-Path $TestCache)) {
#     throw "Missing test cache: $TestCache"
# }

# Write-Host "====================================================="
# Write-Host "Running CivilComments MLP auditor experiment"
# Write-Host "Three optimization seeds; paired within each seed"
# Write-Host "====================================================="

# & $PythonPath `
#   ".\run_labelonly_rebuttal_experiments.py" `
#   --data_root ".\data" `
#   --datasets "civilcomments" `
#   --backbone "all_mpnet_base_v2" `
#   --pretrained 1 `
#   --max_train 0 `
#   --max_test 0 `
#   --fast 0 `
#   --q_grid "1,2,4,8,16,32" `
#   --q_eval 8 `
#   --probes "linear,quadratic,mlp" `
#   --do_stage1_equivalence 0 `
#   --do_safe_overlap 0 `
#   --do_quotient_leakage 1 `
#   --do_frontier_budget 0 `
#   --do_alt_probes 0 `
#   --do_sensitivity 0 `
#   --num_avg 8 `
#   --seed_list "42,123,3407" `
#   --outdir ".\runs_labelonly_rebuttal_text_mlp"

# if ($LASTEXITCODE -ne 0) {
#     throw "CivilComments MLP auditor experiment failed with exit code $LASTEXITCODE"
# }

# Write-Host "====================================================="
# Write-Host "CivilComments MLP auditor experiment completed."
# Write-Host "Outputs:"
# Write-Host "$ScriptDir\runs_labelonly_rebuttal_text_mlp"
# Write-Host "====================================================="


# powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_labelonly_rebuttal_frozen.ps1"




#——————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————

# param(
#     [string]$PythonPath = " "
# )

# $ErrorActionPreference = "Stop"

# $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Set-Location $ScriptDir

# $CoreRoot = ".\runs_labelonly_rebuttal_text_core"
# $MlpRoot = ".\runs_labelonly_rebuttal_text_mlp"

# if (-not (Test-Path $CoreRoot)) {
#     throw "Missing core output root: $CoreRoot"
# }

# if (-not (Test-Path $MlpRoot)) {
#     throw "Missing MLP output root: $MlpRoot"
# }

# $CoreRun = Get-ChildItem `
#   -Path $CoreRoot `
#   -Directory `
#   -Filter "rebuttal_*" |
#   Sort-Object LastWriteTime -Descending |
#   Select-Object -First 1

# $MlpRun = Get-ChildItem `
#   -Path $MlpRoot `
#   -Directory `
#   -Filter "rebuttal_*" | 
#   Sort-Object LastWriteTime -Descending |
#   Select-Object -First 1

# if ($null -eq $CoreRun) {
#     throw "No rebuttal_* directory found under $CoreRoot"
# }

# if ($null -eq $MlpRun) {
#     throw "No rebuttal_* directory found under $MlpRoot"
# }

# $CoreCsv = Join-Path `
#   $CoreRun.FullName `
#   "all_rebuttal_records.csv"

# $MlpCsv = Join-Path `
#   $MlpRun.FullName `
#   "all_rebuttal_records.csv"

# if (-not (Test-Path $CoreCsv)) {
#     throw "Missing core records CSV: $CoreCsv"
# }

# if (-not (Test-Path $MlpCsv)) {
#     throw "Missing MLP records CSV: $MlpCsv"
# }

# Write-Host "====================================================="
# Write-Host "Summarizing CivilComments rebuttal experiment"
# Write-Host "Core records: $CoreCsv"
# Write-Host "MLP records:  $MlpCsv"
# Write-Host "====================================================="

# & $PythonPath `
#   ".\summarize_rebuttal_results.py" `
#   --core_roots "$CoreCsv" `
#   --mlp_roots "$MlpCsv" `
#   --q_eval 8 `
#   --outdir ".\rebuttal_text_summary"

# if ($LASTEXITCODE -ne 0) {
#     throw "CivilComments summary failed with exit code $LASTEXITCODE"
# }

# Write-Host "====================================================="
# Write-Host "CivilComments summary completed."
# Write-Host "Summary directory:"
# Write-Host "$ScriptDir\rebuttal_text_summary"
# Write-Host "====================================================="

# powershell -NoProfile -ExecutionPolicy Bypass `
#   -File ".\run_labelonly_rebuttal_frozen.ps1"