# Docker runtime

Run these commands from the Nutrition AI repository root in PowerShell.
Docker Desktop must be running with Linux containers.

The image uses Python 3.12, CPU-only PyTorch, runtime requirements, application
code, knowledge documents, and the bundled RAG assets. Developer dependencies,
local environments, credentials, tests, and evaluation output are excluded.

## Build

```powershell
docker version
Test-Path .\artifacts\rag\faiss_index.bin
Test-Path .\artifacts\rag\chunks.json
docker build --platform linux/amd64 --progress plain -t nutrition-ai:local .
```

The two `Test-Path` commands must return `True`. Model weights download during
build without Groq or USDA keys. BuildKit caches the downloads across builds;
the Hub cache is also copied into the image, preserving relative symlinks.

The build checks the CPU wheel, RAG row count and embedding dimension, then
verifies that model groups load offline. These checks do not prove that the
index and chunks came from the same build; keep the existing matching pair.

Runtime model loading remains lazy and process-local. Offline mode prevents
Hugging Face model downloads at runtime; Groq and USDA still use the network.

## Run locally

Stop the previous Uvicorn process occupying host port 8001.
Keep `NUTRITION_AI_RAG_DIR` empty in `.env` to use the bundled assets.
The Docker environment file should use plain `NAME=value` entries; unlike
python-dotenv, it does not interpret shell-style variable interpolation.

```powershell
docker run -d --init --name nutrition-ai --env-file .env -p 127.0.0.1:8001:7860 nutrition-ai:local
docker logs --tail 60 nutrition-ai
```

Credentials enter the container at runtime and are not copied into the image.
The service listens on container port 7860, exposed on host loopback port 8001.
The local backend can keep `NUTRITION_AI_BASE_URL=http://127.0.0.1:8001`.
This local setup assumes the backend is running on the host, outside Docker.

## Verify

After the logs show application startup complete:

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/health"
Start-Process "http://127.0.0.1:8001/docs"
```

The health endpoint and Docker health check report process liveness only.
Docker may initially report `starting` before its first health check.

```powershell
docker inspect --format '{{.State.Health.Status}}' nutrition-ai
```

Try these separate Swagger requests with valid provider credentials:

```json
{"message": "Hello"}
```

```json
{"message": "How many calories are in 100 grams of cooked white rice?"}
```

```json
{"message": "What should I eat before a workout?"}
```

```json
{"message": "Suggest a 600-calorie lunch"}
```

Review the HTTP status, processing status, and meal validation metadata.
A 200 response alone does not establish valid nutrition or a valid meal.
Provider errors retain the existing structured error contract.

## Rebuild after changes

The running container uses its image snapshot, not the host source folder.
Rebuild and replace it to apply source or environment-file changes:

```powershell
docker build --platform linux/amd64 --progress plain -t nutrition-ai:local .
docker stop nutrition-ai
docker rm nutrition-ai
docker run -d --init --name nutrition-ai --env-file .env -p 127.0.0.1:8001:7860 nutrition-ai:local
```

## Hosted deployment

For Hugging Face ZeroGPU, use the headless Gradio entry point and configuration
in [runtime modes](runtime_modes.md). This Docker image is the CPU deployment
profile; ZeroGPU does not run this Dockerfile.

Select the hosting platform after local container verification. Configure
Groq and USDA keys using the platform's runtime secrets and set the backend's
AI base URL to the hosted HTTPS address.

Service authentication and request rate limiting are deferred future work.
The current chat endpoint does not require a service key.
Accounts, conversations, profile storage, and meal history remain backend
responsibilities. This image contains no conversation database.