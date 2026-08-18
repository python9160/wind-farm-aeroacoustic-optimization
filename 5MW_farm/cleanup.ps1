[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [Parameter()]
    [switch]$Force
)

# Files/folders we want to keep for OpenFAST to function
$AllowedFiles = @(
  "Airfoils"
  "DISCON.dll"
  "90m_08mps.bts"
  "FAST.Farm.fstf"
  "base.fstf"
  "base_aa.dat"
  "base_ad.dat"
  "FFDB_D100_512x512x64.u"
  "FFDB_D100_512x512x64.v"
  "FFDB_D100_512x512x64.w"
  "FFTest_WT1.fst"
  "FFTest_WT2.fst"
  "base.fst"
  "IW_WT.dat"
  "IW.dat"
  "NRELOffshrBsline5MW_AeroDyn_blade.dat"
  "NRELOffshrBsline5MW_Blade.dat"
  "NRELOffshrBsline5MW_Onshore_ElastoDyn_8mps.dat"
  "NRELOffshrBsline5MW_Onshore_ElastoDyn_Tower.dat"
  "NRELOffshrBsline5MW_Onshore_ServoDyn.dat"
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