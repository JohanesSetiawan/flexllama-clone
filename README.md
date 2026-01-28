# RouterModelCustom

<div align="center">

**A robust model router and manager for Local LLM inference**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

RouterModelCustom is middleware between client applications and `llama-server` instances, providing **dynamic model loading**, **resource management**, **request queuing**, and an **OpenAI-compatible API**.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Starting the Server](#starting-the-server)
  - [Making Requests](#making-requests)
  - [Priority Queue](#priority-queue)
  - [Streaming Responses](#streaming-responses)
- [Monitoring](#monitoring)
  - [Quick Start (Docker)](#quick-start-docker)
  - [Access Dashboards](#access-dashboards)
  - [Prometheus Metrics](#prometheus-metrics)
- [API Reference](#api-reference)
  - [OpenAI Compatible](#openai-compatible)
  - [Management](#management)
  - [Monitoring Endpoints](#monitoring-endpoints)
  - [Realtime Status (SSE)](#realtime-status-sse)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Features

| Feature                         | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| 🔄 **Dynamic Model Management** | Auto-load/unload models based on demand and VRAM availability       |
| ⚡ **Priority Queue System**    | HIGH/NORMAL/LOW priority with heap-based scheduling                 |
| 🎯 **OpenAI Compatible API**    | Drop-in replacement for `/v1/chat/completions` and `/v1/embeddings` |
| 📊 **Prometheus + Grafana**     | Built-in monitoring with pre-configured dashboards                  |
| 🔥 **Model Preloading**         | Warmup models on startup with `preload_models: ["*"]`               |
| 💾 **VRAM Management**          | Real-time tracking and guards to prevent OOM                        |
| 🏥 **Health Monitoring**        | Auto-restart crashed models, health checks every 30s                |
| 🌊 **Streaming Support**        | Server-Sent Events for both inference and status updates            |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    POST /v1/chat/completions
                         {model: "alias"}
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RouterModelCustom (FastAPI)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Queue    │  │ Warmup   │  │ VRAM     │  │ Health           │ │
│  │ Manager  │  │ Manager  │  │ Tracker  │  │ Monitor          │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │             │                  │           │
│       └─────────────┴─────────────┴──────────────────┘           │
│                             │                                    │
│                    ┌────────┴────────┐                           │
│                    │  Model Manager  │                           │
│                    └────────┬────────┘                           │
└─────────────────────────────┼───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ llama-server│     │ llama-server│     │ llama-server│
   │   :8085     │     │   :8086     │     │   :8087     │
   │  (Model A)  │     │  (Model B)  │     │  (Model C)  │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                      ┌───────┴───────┐
                      │   GPU VRAM    │
                      └───────────────┘
```

---

## Getting Started

### Prerequisites

| Requirement    | Version  | Notes                               |
| -------------- | -------- | ----------------------------------- |
| **Python**     | 3.10+    | Required                            |
| **llama.cpp**  | Latest   | Specifically `llama-server` binary  |
| **NVIDIA GPU** | Optional | For GPU acceleration                |
| **Docker**     | Optional | For Prometheus + Grafana monitoring |

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/JohanesSetiawan/flexllama-clone.git
cd flexllama-clone
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**

- `fastapi[standard]` - Web framework
- `httpx[http2]` - Async HTTP client
- `pydantic` - Data validation
- `uvicorn` - ASGI server
- `pynvml` - NVIDIA GPU monitoring
- `aiohttp` - Async HTTP for SSE
- `prometheus-client` - Metrics exposition

#### 4. Prepare llama-server

Download or compile `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
# Example: Download pre-built binary (check llama.cpp releases)
# OR compile from source:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make llama-server GGML_CUDA=1  # With CUDA support
```

### Configuration

#### 1. Create Config File

```bash
cp .example.config.json config.json
```

#### 2. Edit Configuration

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "system": {
    "llama_server_path": "/path/to/llama-server",
    "base_models_path": "/path/to/models",
    "max_concurrent_models": 3,
    "preload_models": ["*"],
    "enable_idle_timeout": false,
    "gpu_devices": [0],
    "default_parallel": 1
  },
  "models": {
    "qwen-7b": {
      "model_path": "qwen-7b.gguf",
      "flags": [
        "-ngl",
        "99",
        "--ctx-size",
        "8192",
        "--flash-attn",
        "on",
        "--mlock",
        "--jinja"
      ]
    },
    "llama-3b": {
      "model_path": "llama-3b.gguf",
      "flags": [
        "-ngl",
        "80",
        "--ctx-size",
        "4096",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--flash-attn",
        "on"
      ]
    },
    "embedding-model": {
      "model_path": "embedding.gguf",
      "flags": ["-ngl", "99", "--ctx-size", "4096", "--embedding"]
    }
  }
}
```

> **TIP:** Use `"preload_models": ["*"]` to load ALL models on startup. Flags are passed directly to `llama-server` - run `llama-server --help` for all options.

#### 3. (Optional) Enable API Key Authentication

```bash
cp .env.example .env
# Edit .env and set API_KEY
```

If `API_KEY` is set, requests must include `Authorization: Bearer <key>` header.

#### 4. Docker Deployment (Recommended)

**Auto-switching Configuration:**

The application automatically detects the environment and uses the appropriate `llama-server` path:

- **Local Development**: Uses path from `konfig.json` (e.g., `/home/user/llama.cpp/build/bin/llama-server`)
- **Docker Container**: Automatically uses `/app/llama-server` from the base image

**Build and Run:**

```bash
# Build Docker image
docker build -t router-model:latest .

# Run container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config.json:/app/config.json \
  router-model:latest
```

**Environment Variable Override:**

You can override the llama-server path at runtime:

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e LLAMA_SERVER_PATH=/custom/path/llama-server \
  -e BASE_MODELS_PATH=/custom/models \
  -v $(pwd)/models:/app/models \
  router-model:latest
```

**Docker Compose:**

```yaml
version: '3.8'
services:
  router:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLAMA_SERVER_PATH=/app/llama-server  # Auto-set by Dockerfile
      - BASE_MODELS_PATH=/app/models
    volumes:
      - ./models:/app/models
      - ./config.json:/app/config.json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

> **Note**: The Dockerfile uses `ghcr.io/ggml-org/llama.cpp:server-cuda` as base image, which includes `llama-server` at `/app/llama-server`. No manual llama.cpp installation needed!

---

## Usage

### Starting the Server

**Original version (main branch):**

```bash
python run.py
```

**Refactored version (refactor branch):**

```bash
python run_refactor.py
```

The server starts on the configured host:port (default: `http://0.0.0.0:8000`).

### Making Requests

#### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

#### Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embedding-model",
    "input": "The quick brown fox"
  }'
```

#### List Available Models

```bash
curl http://localhost:8000/v1/models
```

### Priority Queue

Set request priority via header:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Request-Priority: high" \
  -d '{"model": "qwen-7b", "messages": [...]}'
```

| Priority | Value | Use Case                           |
| -------- | ----- | ---------------------------------- |
| `high`   | 1     | Real-time chat, critical requests  |
| `normal` | 2     | Default for most requests          |
| `low`    | 3     | Batch processing, background tasks |

### Streaming Responses

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

---

## Monitoring

### Quick Start (Docker)

Start Prometheus + Grafana with one command:

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

This will:

- Start **Prometheus** on port `9090`
- Start **Grafana** on port `3000`
- Auto-configure data sources and dashboards

### Access Dashboards

| Service        | URL                   | Credentials       |
| -------------- | --------------------- | ----------------- |
| **Grafana**    | http://localhost:3000 | `admin` / `admin` |
| **Prometheus** | http://localhost:9090 | -                 |

### Prometheus Metrics

The router exposes metrics at `GET /metrics`:

```
# HELP router_requests_total Total requests processed
# TYPE router_requests_total counter
router_requests_total{model="qwen-7b",status="success"} 1234

# HELP router_model_vram_bytes VRAM usage per model
# TYPE router_model_vram_bytes gauge
router_model_vram_bytes{model="qwen-7b"} 4294967296

# HELP router_queue_depth Current queue depth per model
# TYPE router_queue_depth gauge
router_queue_depth{model="qwen-7b"} 5
```

### Stop Monitoring

```bash
docker compose -f docker-compose.monitoring.yml down
```

---

## API Reference

### OpenAI Compatible

| Endpoint               | Method | Description                           |
| ---------------------- | ------ | ------------------------------------- |
| `/v1/chat/completions` | POST   | Chat completion (streaming supported) |
| `/v1/embeddings`       | POST   | Generate embeddings                   |
| `/v1/models`           | GET    | List available models                 |

### Management

| Endpoint                    | Method | Description                                        |
| --------------------------- | ------ | -------------------------------------------------- |
| `/v1/models/eject`          | POST   | Unload model from VRAM. Body: `{"model": "alias"}` |
| `/v1/models/{alias}/reset`  | POST   | Reset failed model status                          |
| `/v1/models/{alias}/status` | GET    | Get loading status of specific model               |
| `/v1/models/failed`         | GET    | List models that failed to load                    |

### Monitoring Endpoints

| Endpoint                | Method | Description                    |
| ----------------------- | ------ | ------------------------------ |
| `/health`               | GET    | Server health check            |
| `/metrics`              | GET    | Prometheus metrics             |
| `/metrics/stream`       | GET    | SSE metrics stream             |
| `/vram`                 | GET    | VRAM usage report              |
| `/v1/queue/stats`       | GET    | Queue statistics               |
| `/v1/telemetry/summary` | GET    | Request telemetry              |
| `/v1/health/models`     | GET    | Health status of loaded models |

### Realtime Status (SSE)

#### Get All Model Status

```bash
curl http://localhost:8000/v1/models/status
```

Response:

```json
{
  "server": { "status": "ready" },
  "models": {
    "qwen-7b": { "status": "ready", "port": 8085, "vram_mb": 4096 },
    "llama-3b": { "status": "off" }
  },
  "summary": { "ready": 1, "off": 1 }
}
```

#### Stream Status Updates

```javascript
const eventSource = new EventSource("/v1/models/status/stream");

eventSource.addEventListener("full_status", (e) => {
  console.log("Initial:", JSON.parse(e.data));
});

eventSource.addEventListener("model_update", (e) => {
  const data = JSON.parse(e.data);
  console.log(`Model ${data.alias}: ${data.status}`);
});
```

---

## Configuration Reference

### Environment Variables

The application supports environment variable overrides for flexible deployment:

| Variable              | Description                                                                 | Example                              |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------ |
| `LLAMA_SERVER_PATH`   | Override llama-server binary path (takes priority over config file)        | `/app/llama-server`                  |
| `BASE_MODELS_PATH`    | Override base models directory (takes priority over config file)           | `/app/models`                        |
| `CONFIG_PATH`         | Override config file path                                                   | `/app/config.json`                   |
| `API_KEY`             | Enable API key authentication                                               | `your-secret-api-key`                |

**Usage Priority:**
1. Environment variables (highest priority)
2. Config file values
3. Default values (lowest priority)

**Example:**
```bash
# Local development - uses konfig.json
python run_refactor.py

# Docker - ENV variables auto-set
docker run -e LLAMA_SERVER_PATH=/app/llama-server router-model:latest

# Custom override
export LLAMA_SERVER_PATH=/custom/path/llama-server
python run_refactor.py
```

### System Configuration (`system`)

| Key                        | Type   | Default | Description                                                                                                             |
| -------------------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `llama_server_path`        | string | -       | Path to `llama-server` binary. **Priority**: ENV `LLAMA_SERVER_PATH` > config value. Docker auto-sets to `/app/llama-server` |
| `base_models_path`         | string | -       | Base directory for model files. **Priority**: ENV `BASE_MODELS_PATH` > config value. If set, `model_path` can be relative |
| `max_concurrent_models`    | int    | 3       | Maximum number of models that can be loaded into VRAM at the same time                                                  |
| `preload_models`           | list   | `[]`    | List of model aliases to load on startup. Use `["*"]` to preload all models defined in config                           |
| `enable_idle_timeout`      | bool   | true    | When enabled, automatically unloads models from VRAM after being idle. Set to `false` to keep models loaded permanently |
| `idle_timeout_sec`         | int    | 300     | Time in seconds a model can remain unused before being unloaded from VRAM                                               |
| `request_timeout_sec`      | int    | 120     | Maximum time in seconds to wait for a response from llama-server before timing out                                      |
| `gpu_devices`              | list   | `[0]`   | List of GPU device indices to use for model inference (e.g., `[0]` for first GPU)                                       |
| `default_parallel`         | int    | 1       | Default parallel request slots (override per-model with `--parallel` flag)                                              |
| `max_queue_size_per_model` | int    | 500     | Maximum number of requests that can wait in queue per model before rejecting new requests                               |
| `vram_multiplier`          | float  | 1.1     | Multiplier applied to model file size when estimating VRAM requirements (1.1 = 10% buffer)                              |

### Model Configuration (`models.{alias}`)

| Key          | Type   | Default  | Description                                                                          |
| ------------ | ------ | -------- | ------------------------------------------------------------------------------------ |
| `model_path` | string | Required | Path to GGUF model file (absolute or relative to `base_models_path`)                 |
| `flags`      | array  | `[]`     | CLI flags passed directly to llama-server. Run `llama-server --help` for all options |

#### Common Flags

| Flag                               | Example                    | Description              |
| ---------------------------------- | -------------------------- | ------------------------ |
| `-ngl`, `--n-gpu-layers`           | `"-ngl", "99"`             | GPU layers (99 = all)    |
| `--ctx-size`, `-c`                 | `"--ctx-size", "4096"`     | Context window size      |
| `--flash-attn`, `-fa`              | `"--flash-attn", "on"`     | Flash Attention mode     |
| `--mlock`                          | `"--mlock"`                | Lock model in RAM        |
| `--embedding`                      | `"--embedding"`            | Enable embedding mode    |
| `--cache-type-k`, `--cache-type-v` | `"--cache-type-k", "q4_0"` | KV cache quantization    |
| `--parallel`                       | `"--parallel", "2"`        | Concurrent request slots |
| `--jinja`                          | `"--jinja"`                | Enable Jinja templates   |

---

## Project Structure

### Original Structure (app/)

```
.
├── app/
│   ├── core/
│   │   ├── config.py          # Configuration loading & validation
│   │   ├── manager.py         # RunnerProcess & ModelManager
│   │   ├── queue.py           # Priority queue system
│   │   ├── warmup.py          # Model preloading
│   │   ├── vram_tracker.py    # GPU VRAM monitoring
│   │   ├── health_monitor.py  # Health checks
│   │   ├── prometheus_metrics.py  # Metrics collection
│   │   └── ...
│   └── main.py                # FastAPI application
├── run.py                     # Server entry point
└── ...
```

### Refactored Structure (app_refactor/)

```
.
├── app_refactor/
│   ├── controllers/           # HTTP endpoints (MVC pattern)
│   │   ├── health_controller.py
│   │   ├── inference_controller.py
│   │   ├── model_controller.py
│   │   ├── status_controller.py
│   │   └── vram_controller.py
│   ├── services/              # Business logic (Service layer)
│   │   ├── proxy_service.py
│   │   ├── embeddings_service.py
│   │   ├── health_service.py
│   │   ├── warmup_service.py
│   │   ├── vram_service.py
│   │   ├── telemetry_service.py
│   │   └── metrics_service.py
│   ├── core/                  # Core infrastructure
│   ├── lifecycle/             # Startup/Shutdown management
│   ├── schemas/               # Pydantic models
│   ├── middlewares/           # HTTP middlewares
│   ├── tasks/                 # Background tasks
│   ├── utils/                 # Utilities
│   ├── routes.py              # Centralized route registry
│   ├── main.py                # FastAPI application factory
│   └── status_server.py       # Lightweight status server (aiohttp)
├── run_refactor.py            # Refactored entry point
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── .example.config.json
├── docker-compose.monitoring.yml
├── requirements.txt
└── README.md
```

---

## Technologies Used

| Category           | Technologies                   |
| ------------------ | ------------------------------ |
| **Backend**        | Python 3.10+, FastAPI, Uvicorn |
| **LLM Inference**  | llama.cpp (llama-server)       |
| **GPU Monitoring** | pynvml (NVIDIA)                |
| **HTTP Client**    | httpx (async)                  |
| **Queue**          | heapq (priority heap)          |
| **Monitoring**     | Prometheus, Grafana            |
| **Validation**     | Pydantic                       |

---

<div align="center">

**Made with ❤️ for the local LLM community**

</div>
