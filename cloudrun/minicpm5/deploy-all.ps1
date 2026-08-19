# Build MiniCPM5 once, then deploy cpu1 / cpu2 / cpu4 in parallel.
# From repo root:  powershell -File cloudrun/minicpm5/deploy-all.ps1
# Skip rebuild:    powershell -File cloudrun/minicpm5/deploy-all.ps1 -SkipBuild

param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$PROJECT = (gcloud config get-value project).Trim()
$REGION  = "europe-west1"
$IMAGE   = "$REGION-docker.pkg.dev/${PROJECT}/llm/minicpm5-1b:cpu"

Write-Host "PROJECT=$PROJECT"
Write-Host "IMAGE=$IMAGE"

if (-not $SkipBuild) {
    Write-Host "`n=== Cloud Build ==="
    gcloud builds submit `
        --config cloudrun/minicpm5/cloudbuild.yaml `
        --substitutions=_IMAGE=$IMAGE `
        .
}

$common = @(
    "--image=$IMAGE",
    "--region=$REGION",
    "--timeout=300",
    "--concurrency=1",
    "--max-instances=1",
    "--min-instances=0",
    "--cpu-boost",
    "--no-cpu-throttling",
    "--allow-unauthenticated",
    "--quiet"
)

$jobs = @(
    @{
        Name = "minicpm5-cpu1"
        Extra = @(
            "--cpu=1", "--memory=1Gi",
            "--set-env-vars=N_THREADS=1,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0"
        )
    },
    @{
        Name = "minicpm5-cpu2"
        Extra = @(
            "--cpu=2", "--memory=1Gi",
            "--set-env-vars=N_THREADS=2,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0"
        )
    },
    @{
        Name = "minicpm5-cpu4"
        Extra = @(
            "--cpu=4", "--memory=2Gi",
            "--set-env-vars=N_THREADS=4,N_CTX=2048,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0"
        )
    }
)

$gcloudCmd = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path -LiteralPath $gcloudCmd)) {
    $found = Get-Command gcloud.cmd -ErrorAction SilentlyContinue
    if ($found) { $gcloudCmd = $found.Source } else { throw "gcloud.cmd not found" }
}

Write-Host "`n=== Deploy 3 services in parallel ==="
$logDir = Join-Path $env:TEMP "minicpm5-cloudrun"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$running = foreach ($j in $jobs) {
    $gcloudArgList = @("run", "deploy", $j.Name) + $common + $j.Extra
    Write-Host ("starting " + $j.Name)
    Start-Process -FilePath $gcloudCmd -ArgumentList $gcloudArgList -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "$($j.Name).log") `
        -RedirectStandardError (Join-Path $logDir "$($j.Name).err")
}

$running | Wait-Process

foreach ($j in $jobs) {
    Write-Host "`n----- $($j.Name) -----"
    Get-Content (Join-Path $logDir "$($j.Name).err") -ErrorAction SilentlyContinue
    Get-Content (Join-Path $logDir "$($j.Name).log") -ErrorAction SilentlyContinue
}

Write-Host "`n=== URLs ==="
gcloud run services list --region=$REGION --filter="metadata.name:minicpm5-"
