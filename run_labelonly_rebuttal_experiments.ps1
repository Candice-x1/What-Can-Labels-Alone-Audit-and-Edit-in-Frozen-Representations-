param(
    [string]$DataRoot = ".\data",
    [string]$Datasets = "celeba,waterbirds,pacs",
    [string]$Backbone = "resnet18",
    [int]$Pretrained = 1,

    [int]$Fast = 0,
    [int]$MaxTrain = 0, 
    [int]$MaxTest = 0,

    [int]$NumAvg = 8,
    [int]$SensitivityNumAvg = 4,

    [string]$QGrid = "1,2,4,8,16,32",
    [int]$QEval = 8,

    [string]$Probes = "linear,quadratic,mlp",
    [string]$AccTols = "0.005,0.01",
    [string]$NllTols = "0.01,0.02",

    [string]$RGrid = "auto",
    [string]$AlphaGrid = "0,0.05,0.10,0.20,0.50",

    [int]$DoStage1Equivalence = 1,
    [int]$DoSafeOverlap = 1,
    [int]$DoQuotientLeakage = 1,
    [int]$DoFrontierBudget = 1,
    [int]$DoAltProbes = 1,
    [int]$DoSensitivity = 1,

    [int]$Seed = 0,
    [string]$SeedList = "42,123,3407,2024,2025,7,17,29,314,2718",

    [string]$OutDir = "runs_labelonly_rebuttal",
    [string]$PyScript = ".\run_labelonly_rebuttal_experiments.py",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if ($PythonPath -eq "") {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Could not find python in PATH. Please pass -PythonPath explicitly."
    }
    $PythonPath = $cmd.Source
}

Write-Host "====================================================="
Write-Host "Running label-only rebuttal experiments"
Write-Host "DataRoot              = $DataRoot"
Write-Host "Datasets              = $Datasets"
Write-Host "Backbone              = $Backbone"
Write-Host "Pretrained            = $Pretrained"
Write-Host "Fast                  = $Fast"
Write-Host "MaxTrain              = $MaxTrain"
Write-Host "MaxTest               = $MaxTest"
Write-Host "NumAvg                = $NumAvg"
Write-Host "SensitivityNumAvg     = $SensitivityNumAvg"
Write-Host "QGrid                 = $QGrid"
Write-Host "QEval                 = $QEval"
Write-Host "Probes                = $Probes"
Write-Host "AccTols               = $AccTols"
Write-Host "NllTols               = $NllTols"
Write-Host "RGrid                 = $RGrid"
Write-Host "AlphaGrid             = $AlphaGrid"
Write-Host "DoStage1Equivalence   = $DoStage1Equivalence"
Write-Host "DoSafeOverlap         = $DoSafeOverlap"
Write-Host "DoQuotientLeakage     = $DoQuotientLeakage"
Write-Host "DoFrontierBudget      = $DoFrontierBudget"
Write-Host "DoAltProbes           = $DoAltProbes"
Write-Host "DoSensitivity         = $DoSensitivity"
Write-Host "Seed                  = $Seed"
Write-Host "SeedList              = $SeedList"
Write-Host "OutDir                = $OutDir"
Write-Host "PyScript              = $PyScript"
Write-Host "PythonPath            = $PythonPath"
Write-Host "====================================================="

if (-not (Test-Path $PyScript)) {
    throw "Python script not found: $PyScript"
}

& $PythonPath $PyScript   --data_root $DataRoot   --datasets $Datasets   --backbone $Backbone   --pretrained $Pretrained   --fast $Fast   --max_train $MaxTrain   --max_test $MaxTest   --num_avg $NumAvg   --sensitivity_num_avg $SensitivityNumAvg   --q_grid $QGrid   --q_eval $QEval   --probes $Probes   --acc_tols $AccTols   --nll_tols $NllTols   --r_grid $RGrid   --alpha_grid $AlphaGrid   --do_stage1_equivalence $DoStage1Equivalence   --do_safe_overlap $DoSafeOverlap   --do_quotient_leakage $DoQuotientLeakage   --do_frontier_budget $DoFrontierBudget   --do_alt_probes $DoAltProbes   --do_sensitivity $DoSensitivity   --seed $Seed   --seed_list $SeedList   --outdir $OutDir

if ($LASTEXITCODE -ne 0) {
    throw "Python script failed with exit code $LASTEXITCODE"
}

Write-Host "====================================================="
Write-Host "Finished rebuttal experiments."
Write-Host "Outputs are under: $OutDir"
Write-Host "====================================================="




#Resnet18
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_rebuttal_experiments.ps1 -Backbone resnet18 -Datasets "celeba,waterbirds,pacs" -Fast 0 -QGrid "1,2,4,8,16,32" -QEval 8 -Probes "linear,quadratic" -DoStage1Equivalence 1 -DoSafeOverlap 1 -DoQuotientLeakage 1 -DoFrontierBudget 1 -DoAltProbes 1 -DoSensitivity 1 -NumAvg 8 -SensitivityNumAvg 4 -SeedList "42,123,3407,2024,2025,7,17,29,314,2718" -OutDir "runs_labelonly_rebuttal_resnet18_core" -PythonPath " "


#CLIP VIT
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_rebuttal_experiments.ps1 -Backbone clip_vit_b32 -Datasets "celeba,waterbirds,pacs" -Fast 0 -QGrid "1,2,4,8,16,32" -QEval 8 -Probes "linear,quadratic" -DoStage1Equivalence 1 -DoSafeOverlap 1 -DoQuotientLeakage 1 -DoFrontierBudget 1 -DoAltProbes 1 -DoSensitivity 1 -NumAvg 8 -SensitivityNumAvg 4 -SeedList "42,123,3407,2024,2025,7,17,29,314,2718" -OutDir "runs_labelonly_rebuttal_clip_core" -PythonPath " "

#MLP probe
#ResNet MLP probe 3 seeds
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_rebuttal_experiments.ps1 -Backbone resnet18 -Datasets "celeba,waterbirds,pacs" -Fast 0 -QGrid "1,2,4,8,16,32" -QEval 8 -Probes "linear,quadratic,mlp" -DoStage1Equivalence 0 -DoSafeOverlap 0 -DoQuotientLeakage 1 -DoFrontierBudget 0 -DoAltProbes 1 -DoSensitivity 0 -NumAvg 8 -SeedList "42,123,3407" -OutDir "runs_labelonly_rebuttal_resnet18_mlp_probe" -PythonPath " "

#CLIP MLP
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_rebuttal_experiments.ps1 -Backbone clip_vit_b32 -Datasets "celeba,waterbirds,pacs" -Fast 0 -QGrid "1,2,4,8,16,32" -QEval 8 -Probes "linear,quadratic,mlp" -DoStage1Equivalence 0 -DoSafeOverlap 0 -DoQuotientLeakage 1 -DoFrontierBudget 0 -DoAltProbes 1 -DoSensitivity 0 -NumAvg 8 -SeedList "42,123,3407" -OutDir "runs_labelonly_rebuttal_clip_mlp_probe" -PythonPath " "



#MLP 10 seeds
#ResNet MLP probe 10 seeds
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_rebuttal_experiments.ps1 -Backbone resnet18 -Datasets "celeba,waterbirds,pacs" -Fast 0 -QGrid "1,2,4,8,16,32" -QEval 8 -Probes "linear,quadratic,mlp" -DoStage1Equivalence 0 -DoSafeOverlap 0 -DoQuotientLeakage 1 -DoFrontierBudget 0 -DoAltProbes 1 -DoSensitivity 0 -NumAvg 8 -SeedList "42,123,3407,0,1,2,3,4,5,2026" -OutDir "runs_labelonly_rebuttal_resnet18_mlp_probe_10seeds" -PythonPath " "

#CLIP MLP
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_rebuttal_experiments.ps1 -Backbone clip_vit_b32 -Datasets "celeba,waterbirds,pacs" -Fast 0 -QGrid "1,2,4,8,16,32" -QEval 8 -Probes "linear,quadratic,mlp" -DoStage1Equivalence 0 -DoSafeOverlap 0 -DoQuotientLeakage 1 -DoFrontierBudget 0 -DoAltProbes 1 -DoSensitivity 0 -NumAvg 8 -SeedList "42,123,3407,0,1,2,3,4,5,2026" -OutDir "runs_labelonly_rebuttal_clip_mlp_probe_10seeds" -PythonPath " "