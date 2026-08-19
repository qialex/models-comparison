# MiniCPM5 1B on Cloud Run (CPU curiosity test)

Same Q4_K_M GGUF as local `:8085` (~657 MB), baked into the image. Local idle RAM is ~750 MB at 2048 ctx, so **1 GiB + 1 vCPU can work** if context stays modest (`N_CTX=1024`). It will be slow.

This is a Cloud Run **service** (HTTP llama.cpp), not a Cloud Function. Functions are for short event handlers; an LLM server needs a long request timeout and a persistent process.

## Cloud Run limits (why not 1/2/3/4 vCPU)

| Wanted | Cloud Run | Memory |
| --- | --- | --- |
| 1 vCPU, 1 GiB | **yes** | 1 GiB |
| 2 vCPU, 1 GiB | **yes** | 1 GiB |
| 3 vCPU | **no** — only 1, 2, 4, 6, 8 | — |
| 4 vCPU | **yes**, but **min 2 GiB** | 2 GiB |

Deploy **three** Cloud Run services (1 / 2 / 4 vCPU). Optional GCE snippet at the bottom if you still want a real 3 vCPU box (min 1.5 GiB on `e2-custom`).

`--concurrency=1` so one request gets the whole machine. `--max-instances=1` keeps cost predictable.

## 0. One-time setup

```powershell
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region europe-west1

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories create llm `
  --repository-format=docker `
  --location=europe-west1 `
  --description="LLM images"

gcloud auth configure-docker europe-west1-docker.pkg.dev
```

Set these once in the same PowerShell session:

```powershell
$PROJECT = (gcloud config get-value project)
$REGION  = "europe-west1"
$IMAGE   = "$REGION-docker.pkg.dev/$PROJECT/llm/minicpm5-1b:cpu"
```

## 1. Build and push (from repo root)

Cloud Build compiles the image in GCP (easier than local Docker on Windows). First build downloads the 657 MB GGUF into the image (~2–5 min).

```powershell
gcloud builds submit `
  --config cloudrun/minicpm5/cloudbuild.yaml `
  --substitutions=_IMAGE=$IMAGE `
  .
```

## 1–2. Build once, deploy all three in parallel

From repo root. One image, three Cloud Run services (`cpu1` / `cpu2` / `cpu4`):

```powershell
powershell -File cloudrun/minicpm5/deploy-all.ps1
```

If Cloud Build already finished:

```powershell
powershell -File cloudrun/minicpm5/deploy-all.ps1 -SkipBuild
```

Or paste this after `$IMAGE` / `$REGION` are set (deploys only, in parallel):

```powershell
$common = "--image=$IMAGE --region=$REGION --timeout=300 --concurrency=1 --max-instances=1 --min-instances=0 --cpu-boost --no-cpu-throttling --allow-unauthenticated --quiet"
$p1 = Start-Process gcloud -PassThru -NoNewWindow -ArgumentList "run deploy minicpm5-cpu1 $common --cpu=1 --memory=1Gi --set-env-vars=N_THREADS=1,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0"
$p2 = Start-Process gcloud -PassThru -NoNewWindow -ArgumentList "run deploy minicpm5-cpu2 $common --cpu=2 --memory=1Gi --set-env-vars=N_THREADS=2,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0"
$p3 = Start-Process gcloud -PassThru -NoNewWindow -ArgumentList "run deploy minicpm5-cpu4 $common --cpu=4 --memory=2Gi --set-env-vars=N_THREADS=4,N_CTX=2048,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0"
$p1,$p2,$p3 | Wait-Process
gcloud run services list --region=$REGION --filter="metadata.name:minicpm5-"
```

## 2. Deploy the three Cloud Run services (one at a time)

```powershell
# 1 vCPU / 1 GiB — the tight curiosity case
gcloud run deploy minicpm5-cpu1 `
  --image=$IMAGE `
  --region=$REGION `
  --cpu=1 `
  --memory=1Gi `
  --timeout=300 `
  --concurrency=1 `
  --max-instances=1 `
  --min-instances=0 `
  --cpu-boost `
  --no-cpu-throttling `
  --set-env-vars=N_THREADS=1,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0 `
  --allow-unauthenticated

# 2 vCPU / 1 GiB
gcloud run deploy minicpm5-cpu2 `
  --image=$IMAGE `
  --region=$REGION `
  --cpu=2 `
  --memory=1Gi `
  --timeout=300 `
  --concurrency=1 `
  --max-instances=1 `
  --min-instances=0 `
  --cpu-boost `
  --no-cpu-throttling `
  --set-env-vars=N_THREADS=2,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0 `
  --allow-unauthenticated

# 4 vCPU / 2 GiB (2 GiB is Cloud Run's minimum for 4 vCPU)
gcloud run deploy minicpm5-cpu4 `
  --image=$IMAGE `
  --region=$REGION `
  --cpu=4 `
  --memory=2Gi `
  --timeout=300 `
  --concurrency=1 `
  --max-instances=1 `
  --min-instances=0 `
  --cpu-boost `
  --no-cpu-throttling `
  --set-env-vars=N_THREADS=4,N_CTX=2048,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,REASONING_BUDGET=0 `
  --allow-unauthenticated
```

If **cpu1** OOMs on startup, bump only that one:

```powershell
gcloud run services update minicpm5-cpu1 --region=$REGION --memory=2Gi
```

## 3. URLs and a timing test

```powershell
gcloud run services list --region=$REGION --filter="metadata.name:minicpm5-"
```

```powershell
$urls = @{
  cpu1 = (gcloud run services describe minicpm5-cpu1 --region=$REGION --format="value(status.url)")
  cpu2 = (gcloud run services describe minicpm5-cpu2 --region=$REGION --format="value(status.url)")
  cpu4 = (gcloud run services describe minicpm5-cpu4 --region=$REGION --format="value(status.url)")
}
$urls

@'
{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0.1}
'@ | Set-Content -Encoding ascii req.json

foreach ($name in @("cpu1","cpu2","cpu4")) {
  $base = $urls[$name]
  Write-Host "`n==== $name  $base ===="
  curl.exe -s -o nul -w "health_http=%{http_code} time=%{time_total}s`n" "$base/health"
  curl.exe -s -w "`nhttp=%{http_code} time=%{time_total}s`n" `
    "$base/v1/chat/completions" `
    -H "Content-Type: application/json" `
    --data-binary "@req.json"
}
```

First request after idle is a **cold start** (load ~657 MB from the image). Run the loop twice and ignore the first timings.

llama.cpp responses include `timings.predicted_per_second` (tok/s). That is the number to compare across vCPU counts.

## 4. Optional: real 3 vCPU on Compute Engine

Cloud Run cannot do 3 vCPU. GCE `e2-custom` can, but **minimum RAM is 0.5 GiB per vCPU**, so 3 vCPU → **1.5 GiB**, not 1 GiB.

```powershell
gcloud compute firewall-rules create allow-minicpm5-8080 --allow=tcp:8080 --target-tags=minicpm5

gcloud compute instances create-with-container minicpm5-cpu3 `
  --zone=europe-west1-b `
  --machine-type=e2-custom-3-1536 `
  --tags=minicpm5 `
  --container-image=$IMAGE `
  --container-env=N_THREADS=3,N_CTX=1024,N_PREDICT=128,N_GPU_LAYERS=0,REASONING=off,PORT=8080

gcloud compute instances describe minicpm5-cpu3 --zone=europe-west1-b --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

Then `http://THE_IP:8080/v1/chat/completions`. Delete when done: `gcloud compute instances delete minicpm5-cpu3 --zone=europe-west1-b`.

## 5. Tear down Cloud Run

```powershell
gcloud run services delete minicpm5-cpu1 --region=$REGION --quiet
gcloud run services delete minicpm5-cpu2 --region=$REGION --quiet
gcloud run services delete minicpm5-cpu4 --region=$REGION --quiet
```

`--min-instances=0` already scales to zero when idle; you still pay a little for Artifact Registry storage until you delete the image.
