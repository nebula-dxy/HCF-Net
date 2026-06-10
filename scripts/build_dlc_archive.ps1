$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$bundleRoot = Join-Path $root "dlc_bundle"
$projectRoot = Join-Path $bundleRoot "gemini"
$zipPath = Join-Path $bundleRoot "gemini_bundle.zip"

if (-not (Test-Path $projectRoot)) {
    Write-Host "Bundle folder not found, preparing it first..."
    python (Join-Path $PSScriptRoot "prepare_dlc_bundle.py")
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path (Join-Path $projectRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Created archive:" $zipPath
