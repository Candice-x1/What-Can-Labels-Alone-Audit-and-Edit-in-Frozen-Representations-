param(  
    [string]$DataRoot = ".\data",  
    [string]$Datasets = "celeba,waterbirds,pacs",  
    [string]$Backbone = "resnet18",  
    [int]$Pretrained = 1,  
  
    [int]$Fast = 0,  
    [int]$DoRandomControl = 0,  
  
    [int]$MaxTrain = 0,  
    [int]$MaxTest = 0,  
  
    [int]$DoFrontier = 1,  
    [int]$DoAbstention = 1,  
    [int]$DoDiagE = 1,  
    [int]$DoSyntheticAppendix = 1,  
  
    [string]$FrontierQs = "1,2,3,5,8,12",  
  
    [int]$DiagEQ = 3,  
    [int]$DiagEPerms = 32,  
    [int]$DiagESplitRepeats = 5,  
  
    [string]$AbstentionFracs = "1.0,0.7,0.4,0.2",  
    [int]$AbstentionRepeats = 5,  
    [double]$AbstentionSupportThresh = 0.45,  
    [double]$AbstentionStabilityThresh = 0.55,  
    [double]$AbstentionMassThresh = 0.01,  
    [double]$AbstentionLabelDamageTol = 0.02,  
  
    [int]$Seed = 0,  
    [int]$NumRuns = 1,  
    [int]$SeedStride = 1,  
    [string]$SeedList = "42,123,3407,2024,2025,7,17,29,314,2718",  
  
    [string]$OutDir = "runs_labelonly_workflow_full_appendix",  
    [string]$PyScript = ".\run_labelonly_workflow_full_v2_appendix.py"  
)  
  
$ErrorActionPreference = "Stop"  
  
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path  
Set-Location $ScriptDir  
  
Write-Host "====================================================="  
Write-Host "Running appendix-complete label-only workflow"  
Write-Host "DataRoot                  = $DataRoot"  
Write-Host "Datasets                  = $Datasets"  
Write-Host "Backbone                  = $Backbone"  
Write-Host "Pretrained                = $Pretrained"  
Write-Host "Fast                      = $Fast"  
Write-Host "DoRandomControl           = $DoRandomControl"  
Write-Host "DoFrontier                = $DoFrontier"  
Write-Host "DoAbstention              = $DoAbstention"  
Write-Host "DoDiagE                   = $DoDiagE"  
Write-Host "DoSyntheticAppendix       = $DoSyntheticAppendix"  
Write-Host "SeedList                  = $SeedList"  
Write-Host "OutDir                    = $OutDir"  
Write-Host "PyScript                  = $PyScript"  
Write-Host "====================================================="  
  
$PythonPath = ""  
  
if (-not (Test-Path $PythonPath)) {  
    throw "Python executable not found: $PythonPath"  
}  
  
if (-not (Test-Path $PyScript)) {  
    throw "Python script not found: $PyScript"  
}  
  
& $PythonPath $PyScript --data_root $DataRoot --datasets $Datasets --backbone $Backbone --pretrained $Pretrained --fast $Fast --do_random_control $DoRandomControl --max_train $MaxTrain --max_test $MaxTest --do_frontier $DoFrontier --do_abstention $DoAbstention --do_diagE $DoDiagE --do_synthetic_appendix $DoSyntheticAppendix --frontier_qs $FrontierQs --diagE_q $DiagEQ --diagE_perms $DiagEPerms --diagE_split_repeats $DiagESplitRepeats --abstention_fracs $AbstentionFracs --abstention_repeats $AbstentionRepeats --abstention_support_thresh $AbstentionSupportThresh --abstention_stability_thresh $AbstentionStabilityThresh --abstention_mass_thresh $AbstentionMassThresh --abstention_label_damage_tol $AbstentionLabelDamageTol --seed $Seed --num_runs $NumRuns --seed_stride $SeedStride --seed_list $SeedList --outdir $OutDir  
  
if ($LASTEXITCODE -ne 0) {  
    throw "Python script failed with exit code $LASTEXITCODE"  
}  
  
Write-Host "====================================================="  
Write-Host "Finished."  
Write-Host "Main summary: $OutDir"  
Write-Host "Appendix aggregate files: $OutDir\multiseed_*\aggregate_appendix"  
Write-Host "Synthetic appendix files: $OutDir\multiseed_*\synthetic_appendix"  
Write-Host "====================================================="  


#CLIP
#powershell -ExecutionPolicy Bypass -File .\run_labelonly_workflow_full_appendix_multiseed.ps1 -Backbone clip_vit_b32  

#Resnet18
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_workflow_full_appendix_multiseed.ps1 -Backbone resnet18  


# two backbone run once
# powershell -ExecutionPolicy Bypass -File .\run_labelonly_workflow_full_appendix_multiseed.ps1 -Backbone "resnet18,clip_vit_b32"  
