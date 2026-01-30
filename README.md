# RouterModelCustom

<div align="center">

**RouterModelCustom is middleware between client applications and `llama-server` instances, providing **dynamic model loading**, **resource management**, **request queuing**, and an **OpenAI-compatible API**.**

</div>

> **Refactored Version Available**: The `refactor` branch features a complete architectural redesign with clean separation of concerns, dependency injection, improved observability, and production-ready features including Redis caching, rate limiting, and enhanced monitoring.

---

## Table of Contents

- [RouterModelCustom](#routermodelcustom)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
    - [Core Features](#core-features)
    - [Refactored Version Features (v2.0)](#refactored-version-features-v20)
  - [Architecture Overview](#architecture-overview)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
      - [1. Clone the Repository](#1-clone-the-repository)
      - [2. Create Virtual Environment](#2-create-virtual-environment)
      - [3. Install Dependencies](#3-install-dependencies)
      - [4. Prepare llama-server](#4-prepare-llama-server)
    - [Configuration](#configuration)
      - [1. Create Config File](#1-create-config-file)
      - [2. Edit Configuration](#2-edit-configuration)
      - [3. (Optional) Redis Configuration for Refactored Version](#3-optional-redis-configuration-for-refactored-version)
      - [4. (Optional) Enable API Key Authentication](#4-optional-enable-api-key-authentication)
      - [5. Docker Deployment (Recommended)](#5-docker-deployment-recommended)
  - [Usage](#usage)
    - [Starting the Server](#starting-the-server)
    - [Making Requests](#making-requests)
      - [Chat Completion](#chat-completion)
      - [Embeddings](#embeddings)
      - [List Available Models](#list-available-models)
    - [Priority Queue](#priority-queue)
    - [Streaming Responses](#streaming-responses)
  - [Monitoring](#monitoring)
    - [Quick Start (Docker)](#quick-start-docker)
    - [Access Dashboards](#access-dashboards)
    - [Prometheus Metrics](#prometheus-metrics)
    - [Stop Monitoring](#stop-monitoring)
  - [API Reference](#api-reference)
    - [OpenAI Compatible](#openai-compatible)
    - [Management](#management)
    - [Monitoring Endpoints](#monitoring-endpoints)
    - [Refactored Version Additional Endpoints (v2.0)](#refactored-version-additional-endpoints-v20)
    - [Realtime Status (SSE)](#realtime-status-sse)
      - [Get All Model Status](#get-all-model-status)
      - [Stream Status Updates](#stream-status-updates)
  - [Configuration Reference](#configuration-reference)
    - [Environment Variables](#environment-variables)
    - [System Configuration (`system`)](#system-configuration-system)
    - [Model Configuration (`models.{alias}`)](#model-configuration-modelsalias)
      - [Common Flags](#common-flags)
  - [Project Structure](#project-structure)
    - [Original Structure (app/)](#original-structure-app)
    - [Refactored Structure (app\_refactor/)](#refactored-structure-app_refactor)
  - [](#)
  - [Technologies Used](#technologies-used)
    - [Core Stack (Available in Main Branch)](#core-stack-available-in-main-branch)
    - [Core Stack + Additional Technologies](#core-stack--additional-technologies)
  - [Version Comparison](#version-comparison)
    - [When to Use Main Branch](#when-to-use-main-branch)
    - [When to Use Refactor Branch](#when-to-use-refactor-branch)
    - [Migration Guide](#migration-guide)

---

## Features

### Core Features

| Feature                         | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| **Dynamic Model Management** | Auto-load/unload models based on demand and VRAM availability       |
| **Priority Queue System**    | HIGH/NORMAL/LOW priority with heap-based scheduling                 |
| **OpenAI Compatible API**    | Drop-in replacement for `/v1/chat/completions` and `/v1/embeddings` |
| **Prometheus + Grafana**     | Built-in monitoring with pre-configured dashboards                  |
| **Model Preloading**         | Warmup models on startup with `preload_models: ["*"]`               |
| **VRAM Management**          | Real-time tracking and guards to prevent OOM                        |
| **Health Monitoring**        | Auto-restart crashed models, health checks every 30s                |
| **Streaming Support**        | Server-Sent Events for both inference and status updates            |

### Refactored Version Features (v2.0)

| Feature                          | Description                                                              |
| -------------------------------- | ------------------------------------------------------------------------ |
| **Clean Architecture**       | MVC pattern with controllers, services, and dependency injection         |
| **Redis Support**             | Semantic caching, Redis-backed queue for distributed deployments        |
| **Rate Limiting**             | Redis-backed rate limiter with per-user quotas                           |
| **Enhanced Telemetry**        | Detailed request tracking with latency histograms and error analytics    |
| **Request Tracking**          | Unique request IDs, correlation tracking, and distributed tracing ready  |
| **Modular Design**            | Clean separation: controllers → services → core infrastructure           |
| **Dependency Injection**      | Centralized container for testability and maintainability                |
| **Production Ready**          | Comprehensive error handling, graceful shutdown, and resource cleanup    |

---

## Architecture Overview

<details>
<summary><h3>Main Branch Architecture</h3></summary>

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

</details>

<details open>
<summary><h3>Refactor Branch Architecture</h3></summary>

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Middleware Stack                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ CORS → Auth → Rate Limit → Size Limit → Telemetry    │  │
│  └─────────────────────────┬─────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Controllers Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Health     │  │  Inference   │  │    Model        │   │
│  │  Controller  │  │  Controller  │  │  Controller     │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────────┘   │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      Services Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Proxy   │  │  Cache   │  │  Queue   │  │  Health  │    │
│  │ Service  │  │ Service  │  │ Service  │  │ Service  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Infrastructure                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │    Model     │  │    Queue     │  │      VRAM       │   │
│  │   Manager    │  │   Manager    │  │    Tracker      │   │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬───────┘   │
└─────────┼──────────────────┼──────────────────────┼─────────┘
          │                  │                      │
          └──────────────────┴──────────────────────┘
                             │
                    llama-server instances
```

**Key Architectural Improvements:**
- **Dependency Injection**: Centralized `AppContainer` for all components
- **Service Layer**: Business logic isolated from HTTP concerns
- **Clean Controllers**: Thin layer handling only HTTP request/response
- **Modular Design**: Each component has single responsibility
- **Testability**: Mock-friendly architecture for unit testing
- **Status Server**: Separate lightweight aiohttp server for real-time status updates

</details>

---

## Getting Started

### Prerequisites

| Requirement    | Version  | Notes                                        |
| -------------- | -------- | -------------------------------------------- |
| **Python**     | 3.10+    | Required                                     |
| **llama.cpp**  | Latest   | Specifically `llama-server` binary           |
| **NVIDIA GPU** | Optional | For GPU acceleration                         |
| **Docker**     | Optional | For Prometheus + Grafana monitoring          |
| **Redis**      | Optional | For caching & rate limiting (refactor only)  |

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

<details open>
<summary><h4>Refactor Branch Only - Redis Configuration</h4></summary>

#### 3. (Optional) Redis Configuration for Refactored Version

The refactored version supports Redis for semantic caching and distributed queue:

```json
{
  "redis": {
    "url": "redis://localhost:6379/0",
    "enable_cache": true,
    "cache_ttl_sec": 3600,
    "enable_queue": false
  },
  "rate_limit": {
    "requests_per_minute": 60,
    "redis_url": "redis://localhost:6379/0"
  }
}
```

**Benefits:**
- **Semantic Caching**: Cache LLM responses to reduce GPU load and latency
- **Distributed Queue**: Share request queue across multiple router instances
- **Rate Limiting**: Protect against abuse with per-user request limits

</details>

#### 4. (Optional) Enable API Key Authentication

```bash
cp .env.example .env
# Edit .env and set API_KEY
```

If `API_KEY` is set, requests must include `Authorization: Bearer <key>` header.

#### 5. Docker Deployment (Recommended)

**Auto-switching Configuration:**

The application automatically detects the environment and uses the appropriate `llama-server` path:

- **Local Development**: Uses path from `config.json` (e.g., `/home/user/llama.cpp/build/bin/llama-server`)
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

<details>
<summary><h4>Main Branch - Starting Server</h4></summary>

```bash
python run.py
```

</details>

<details open>
<summary><h4>Refactor Branch - Starting Server</h4></summary>

```bash
python run_refactor.py
```

</details>

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
| `/v1/telemetry/summary` | GET    | Request telemetry (v2.0 only)  |
| `/v1/health/models`     | GET    | Health status of loaded models |

<details open>
<summary><h3>Refactor Branch - Additional API Endpoints</h3></summary>

### Refactored Version Additional Endpoints (v2.0)

| Endpoint                     | Method | Description                                  |
| ---------------------------- | ------ | -------------------------------------------- |
| `/v1/cache/stats`            | GET    | Redis cache hit/miss statistics              |
| `/v1/cache/clear`            | POST   | Clear semantic cache                         |
| `/v1/telemetry/requests`     | GET    | Detailed request history                     |
| `/v1/telemetry/errors`       | GET    | Error analytics and patterns                 |
| `/v1/models/status/stream`   | GET    | SSE stream for real-time model status        |

</details>
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
| `REDIS_URL`           | Redis connection URL (refactor only)                                        | `redis://localhost:6379/0`           |

**Usage Priority:**
1. Environment variables (highest priority)
2. Config file values
3. Default values (lowest priority)

**Example:**
```bash
# Local development - uses config.json
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

<details>
<summary><h3>Main Branch - Project Structure</h3></summary>

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

</details>

<details open>
<summary><h3>Refactor Branch - Project Structure</h3></summary>

### Refactored Structure (app_refactor/)

```
.
├── app_refactor/
│   ├── controllers/           # HTTP endpoints (MVC pattern)
│   │   ├── health_controller.py     # /health, /v1/health/*
│   │   ├── inference_controller.py  # /v1/chat/completions
│   │   ├── model_controller.py      # /v1/models/*
│   │   ├── metrics_controller.py    # /metrics, /v1/telemetry/*
│   │   ├── status_controller.py     # /v1/models/status
│   │   └── vram_controller.py       # /vram
│   ├── services/              # Business logic layer
│   │   ├── proxy_service.py         # LLM request proxying
│   │   ├── embeddings_service.py    # Embeddings generation
│   │   ├── cache_service.py         # Redis semantic caching
│   │   ├── redis_queue_service.py   # Redis-backed queue
│   │   ├── health_service.py        # Health monitoring
│   │   ├── warmup_service.py        # Model preloading
│   │   ├── vram_service.py          # VRAM tracking
│   │   ├── telemetry_service.py     # Request analytics
│   │   └── metrics_service.py       # Prometheus metrics
│   ├── core/                  # Core infrastructure
│   │   ├── config.py                # Config with env var override
│   │   ├── manager.py               # Model lifecycle manager
│   │   ├── queue.py                 # Priority heap queue
│   │   ├── errors.py                # Custom exceptions
│   │   ├── logging_server.py        # Structured logging
│   │   └── model_status.py          # Status tracking
│   ├── lifecycle/             # Application lifecycle
│   │   ├── startup.py               # Startup initialization
│   │   ├── shutdown.py              # Graceful shutdown
│   │   └── dependencies.py          # Dependency injection
│   ├── schemas/               # Pydantic request/response models
│   ├── middlewares/           # HTTP middleware stack
│   │   ├── auth.py                  # API key authentication
│   │   ├── rate_limit.py            # Redis rate limiter
│   │   ├── telemetry_middleware.py  # Request tracking
│   │   ├── metrics_middleware.py    # Prometheus metrics
│   │   └── limit_request.py         # Request size limiter
│   ├── tasks/                 # Background tasks
│   │   └── status_sync.py           # Status file sync
│   ├── utils/                 # Utility functions
│   │   ├── gguf_parser.py           # GGUF metadata parser
│   │   └── legacy_metrics.py        # Legacy metrics support
│   ├── routes.py              # Centralized route registry
│   ├── main.py                # FastAPI application factory
│   └── status_server.py       # Lightweight aiohttp status server
├── run_refactor.py            # Refactored entry point
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── config.json                # Configuration file
├── docker-compose.monitoring.yml
├── requirements.txt
└── README.md
```

**Key Refactored Features:**
- Clean Architecture: Controllers → Services → Core
- Dependency Injection via AppContainer
- Redis support for caching and distributed queue
- Rate limiting with Redis backend
- Enhanced telemetry and observability
- Separate status server for real-time updates
- Comprehensive error handling and logging
- Production-ready with graceful shutdown
</details>
---

## Technologies Used

<details>
<summary><h3>Main Branch - Core Technologies</h3></summary>

### Core Stack (Available in Main Branch)

| Category           | Technologies                   |
| ------------------ | ------------------------------ |
| **Backend**        | Python 3.10+, FastAPI, Uvicorn |
| **LLM Inference**  | llama.cpp (llama-server)       |
| **GPU Monitoring** | pynvml (NVIDIA)                |
| **HTTP Client**    | httpx (async)                  |
| **Queue**          | heapq (priority heap)          |
| **Monitoring**     | Prometheus, Grafana            |
| **Validation**     | Pydantic                       |

</details>

<details open>
<summary><h3>Refactor Branch - All Technologies</h3></summary>

### Core Stack + Additional Technologies

**All Main Branch technologies PLUS:**

| Category              | Technologies                    |
| --------------------- | ------------------------------- |
| **HTTP Client**       | httpx (async) + aiohttp (SSE)   |
| **Caching**           | Redis (semantic caching)        |
| **Rate Limiting**     | Redis (distributed limiter)     |
| **Status Server**     | aiohttp (lightweight SSE)       |
| **Logging**           | Structured logging with context |
| **Architecture**      | Clean Architecture + DI         |
| **Error Handling**    | Custom exception hierarchy      |

</details>

---

## Version Comparison

<details>
<summary><h3>Main Branch (Original)</h3></summary>

### When to Use Main Branch

**Main Branch** - Use if you need:
- ✅ Simple, straightforward setup
- ✅ No external dependencies (Redis, etc.)
- ✅ Quick prototyping and testing
- ✅ Minimal resource overhead

</details>

<details open>
<summary><h3>Refactor Branch</h3></summary>

### When to Use Refactor Branch

**Refactor Branch** - Use if you need:
- ✅ Production deployment at scale
- ✅ Redis caching for reduced GPU load
- ✅ Rate limiting and API protection
- ✅ Distributed deployment across multiple instances
- ✅ Enhanced observability and debugging
- ✅ Better code maintainability
- ✅ Advanced telemetry and analytics

</details>

### Migration Guide

To migrate from original to refactored version:

1. **Switch Branch:**
   ```bash
   git checkout refactor
   ```

2. **Install Additional Dependencies:**
   ```bash
   pip install -r requirements.txt
   # Optionally install redis-py for caching
   pip install redis
   ```

3. **Update Configuration:**
   - Configuration format is compatible
   - Optionally add Redis and rate_limit sections
   - No breaking changes to existing config

4. **Change Entry Point:**
   ```bash
   # Old: python run.py
   python run_refactor.py
   ```

5. **Optional Redis Setup:**
   ```bash
   docker run -d -p 6379:6379 redis:alpine
   ```

---

<div align="center">

**Made with ❤️**

</div>
