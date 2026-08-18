[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter()]
    [switch]$Force
)

# Files/folders we want to keep for OpenFAST to function
$AllowedFiles = @(
  "NRELOffshrBsline5MW_AeroDyn_blade.dat"
  "Airfoils"
  "WindInflowFile.dat"
  "AA.dat"
  "NRELOffshrBsline5MW_Onshore_ElastoDyn_Tower.dat"
  "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat"
  "AA_AeroDyn.dat"
  "NRELOffshrBsline5MW_Blade.dat"
  "AA_loc.dat"
  "Turbine.fst"
  "NRELOffshrBsline5MW_Onshore_ServoDyn.dat"
  "NRELOffshrBsline5MW_BeamDyn.dat"
  "DISCON.dll"
)

# Kill OpenFAST
Write-Host "Checking OpenFAST status..." -ForegroundColor Yellow

$openfastProcesses = Get-Process -Name "*OpenFAST*" -ErrorAction SilentlyContinue

if ($openfastProcesses) {
    $openfastProcesses | Stop-Process -Force
    Write-Host "OpenFAST stopped." -ForegroundColor Green
} else {
    Write-Host "OpenFAST has already closed." -ForegroundColor Yellow
}

[System.Threading.Thread]::Sleep(500)

$ScriptName = Split-Path $PSCommandPath -Leaf
$AllowedFiles += $ScriptName

# 1. Find all files to be deleted
$CurrentFiles = Get-ChildItem -Path $PSScriptRoot -File
$FilesToDelete = $CurrentFiles | Where-Object { $_.Name -notin $AllowedFiles }

# If there's nothing to delete, exit early
if ($FilesToDelete.Count -eq 0) {
    Write-Host "No files to clean up!" -ForegroundColor Green
    return
}

# 2. Print the list of files to be deleted upfront
Write-Host "The following files are NOT in the allowed list and will be deleted:" -ForegroundColor Cyan
foreach ($File in $FilesToDelete) {
    Write-Host " - $($File.Name)" -ForegroundColor Yellow
}
Write-Host "" # Empty line for spacing

# 3. Process the deletions
foreach ($File in $FilesToDelete) {
    if ($Force -or $PSCmdlet.ShouldProcess($File.FullName, "Delete File")) {
        Remove-Item -Path $File.FullName -Force
        Write-Host "Deleted: $($File.Name)" -ForegroundColor Gray
    }
}

Write-Host "Cleanup complete!" -ForegroundColor Green