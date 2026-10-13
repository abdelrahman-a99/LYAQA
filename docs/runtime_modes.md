# CPU, CUDA and ZeroGPU runtime

## Configuration and installation are separate

`NUTRITION_AI_RUNTIME` is read once per process, before model loading.
Changing it requires a restart. Its default is `cpu`. Unknown values fail with
`runtime_configuration_error`. The setting selects execution behavior; it does
not provision hardware, install drivers or replace the installed PyTorch wheel.

| Runtime | Model device | Hosting | Loading |
| --- | --- | --- | --- |
| cpu | CPU, even if CUDA is available | Direct FastAPI REST | Lazy |
| cuda | Persistent CUDA | Direct FastAPI REST | Lazy |
| zerogpu | Startup CUDA emulation, real CUDA during allocated tasks | Headless Gradio Server and queued model endpoint | Prepared before launch |

CUDA mode requires a working CUDA-enabled PyTorch installation and GPU.
CUDA mode is rejected on ZeroGPU hardware. ZeroGPU mode requires the hosted
scheduler, a supported CUDA-enabled PyTorch build, and `space_app.py`.
Neither mode silently falls back to CPU.

## Local CPU

From the repository root, PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
$env:NUTRITION_AI_RUNTIME = "cpu"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Install `requirements.txt` instead of `requirements-dev.txt` for runtime only.
Installing the CPU wheel first prevents a generic requirements installation
from selecting a CUDA wheel for a CPU machine. The Dockerfile also installs
the CPU wheel first. It is a CPU deployment profile.

## Persistent CUDA

On compatible NVIDIA hardware, install the CUDA build and a compatible driver:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:NUTRITION_AI_RUNTIME = "cuda"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

A CUDA image is a separate deployment profile; changing the CPU Docker
container's environment does not add CUDA libraries or GPU access.

## Hugging Face ZeroGPU

Create a Gradio Space on an eligible account and select ZeroGPU hardware.
Free hosting eligibility currently requires a verified email and an account
older than 30 days. A newly created account cannot immediately use this route.

The README metadata uses `sdk: gradio`, `sdk_version: 6.15.0`, Python 3.12,
`app_file: space_app.py`, and `base_path: /docs`. It launches Gradio Server without
a Gradio user interface. The current REST schemas and error handlers are reused.

Set these Space Variables:

| Name | Value |
| --- | --- |
| NUTRITION_AI_RUNTIME | zerogpu |
| GRADIO_SSR_MODE | false |
| USDA_BASE_URL | https://api.nal.usda.gov/fdc/v1 (optional) |

Set these Secrets:

| Name | Purpose |
| --- | --- |
| GROQ_API_KEY_NLP | Required Groq key; also fallback when optional keys are absent |
| USDA_API_KEY | Required USDA key |
| GROQ_API_KEY_MEALS | Optional valid dedicated Groq key |
| GROQ_API_KEY_RAG | Optional valid dedicated Groq key |
| HF_TOKEN | Required read token belonging to the account whose GPU quota is used |

`SPACE_ID`, `SPACES_ZERO_GPU`, and scheduler settings are injected by Hugging
Face. Do not invent or manually set those hosted scheduler settings.
Leave `NUTRITION_AI_RAG_DIR` unset to use the bundled RAG files.
Do not set `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE` for this deployment: model
files are downloaded during startup. Do not copy the local `.env` into Git.

The currently pinned Gradio release supports `Server` while remaining compatible
with Hugging Face Hub 0.36.2 and Transformers 4.44.2. PyTorch 2.8.0 is shared
between CPU, CUDA and ZeroGPU; the installed wheel still differs by deployment.

### Execution boundary

1. A caller sends the existing JSON request to `POST /v1/chat`.
2. Parsing, Groq calls, USDA HTTP requests, calculations and FAISS search run
   outside the GPU lease.
3. Semantic embeddings, cross-encoder scoring, translation and RAG query
   embedding use `run_inference`.
4. In ZeroGPU mode, the adapter calls the Space's registered `/model_compute`
   Gradio API using the official Python client and `HF_TOKEN`.
5. Gradio queues the model operation. Only its kernel uses `spaces.GPU`.
6. The adapter validates its result and the original REST route returns the
   existing `ChatResponse`, or a controlled HTTP processing error.

This self-call goes through the public HF Space API so HF establishes the quota
identity associated with `HF_TOKEN`. All REST users currently share that account's
quota. No caller-specific HF identity routing has been implemented.

The model endpoint is undocumented in Gradio API listings, but it is not access
controlled. Service authentication and rate limiting remain deferred future work.

### Quotas

GPU quota measures allocated GPU execution, not the total uptime of the Space.
The current free-account allowance is 5 minutes daily, PRO is 40 minutes, and
unauthenticated usage has a separate smaller allowance. Allowance resets 24
hours after the account's first GPU use. The same account shares its allowance
across calls and Spaces. A task's requested duration also needs to fit the
remaining allowance; these small kernels request up to 20 seconds each.

Quota exhaustion rejects GPU work. It does not automatically pause the entire
Space or enable a CPU fallback. Routes that do not need GPU work can continue
subject to the hosting platform's lifecycle. Space sleep is a separate issue.

### Additional processing codes

| Code | HTTP | Retryable |
| --- | --- | --- |
| runtime_configuration_error | 500 | false |
| zerogpu_quota_exceeded | 503 | false |
| zerogpu_unavailable | 503 | true |
| zerogpu_execution_failed | 503 | false |
| zerogpu_invalid_output | 502 | false |
| zerogpu_timeout | 504 | true |
| zerogpu_authentication_failed | 502 | false |

Quota errors should not be replayed immediately. The adapter submits each model
operation once. Its 120-second wait timeout attempts cancellation; cancelling
cannot guarantee that already-running GPU work stops immediately. No CPU retry
or alternate identity is used to bypass quota.

## Assets and verification

Keep the bundled FAISS index and chunks together. Rebuild the index offline
with CPU or persistent CUDA before deployment, not inside ZeroGPU runtime.
FAISS remains on CPU; the embedding model uses the selected device.

```powershell
.\.venv\Scripts\python.exe -m ruff check app evaluation scripts tests space_app.py
.\.venv\Scripts\python.exe -m ruff format --check app evaluation scripts tests space_app.py
.\.venv\Scripts\python.exe -m pytest -q
```

A headless local hosting check can use CPU mode:

```powershell
$env:NUTRITION_AI_RUNTIME = "cpu"
.\.venv\Scripts\python.exe space_app.py
```

It serves `/health`, `/docs`, and `/v1/chat` on localhost port 7860. This checks
the Gradio hosting integration; it does not emulate a real ZeroGPU GPU lease.

Local verification covers the existing tests, runtime failure cases, the REST
error contract, a real headless Gradio server and queued JSON calls, and the
four real models on CPU. Actual CUDA execution, HF quota accounting and hosted
ZeroGPU model compatibility require verification on the target hardware.

Sources:
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/docs/hub/spaces-overview
- https://pypi.org/project/gradio/6.15.0/
- https://pytorch.org/get-started/previous-versions/
