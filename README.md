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
    "max_concurrent_models": 3,
    "preload_models": ["*"],
    "enable_idle_timeout": false,
    "gpu_devices": [0],
    "flash_attention": "on"
  },
  "models": {
    "qwen-7b": {
      "model_path": "/path/to/models/qwen-7b.gguf",
      "params": {
        "n_gpu_layers": 99,
        "n_ctx": 8192
      }
    },
    "llama-3b": {
      "model_path": "/path/to/models/llama-3b.gguf",
      "params": {
        "n_gpu_layers": 99,
        "n_ctx": 4096
      }
    }
  }
}
```

> **TIP:** Use `"preload_models": ["*"]` to load ALL models on startup (ideal for avoiding cold-start latency).

---

## Usage

### Starting the Server

```bash
python run.py
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

### System Configuration (`system`)

| Key                        | Type   | Default | Description                                                                                                             |
| -------------------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `llama_server_path`        | string | -       | Absolute path to the `llama-server` executable binary from llama.cpp                                                    |
| `max_concurrent_models`    | int    | 3       | Maximum number of models that can be loaded into VRAM at the same time                                                  |
| `preload_models`           | list   | `[]`    | List of model aliases to load on startup. Use `["*"]` to preload all models defined in config                           |
| `enable_idle_timeout`      | bool   | true    | When enabled, automatically unloads models from VRAM after being idle. Set to `false` to keep models loaded permanently |
| `idle_timeout_sec`         | int    | 300     | Time in seconds a model can remain unused before being unloaded from VRAM                                               |
| `request_timeout_sec`      | int    | 120     | Maximum time in seconds to wait for a response from llama-server before timing out                                      |
| `gpu_devices`              | list   | `[0]`   | List of GPU device indices to use for model inference (e.g., `[0]` for first GPU)                                       |
| `parallel_requests`        | int    | 1       | Number of concurrent requests each model instance can handle simultaneously                                             |
| `cpu_threads`              | int    | 8       | Number of CPU threads allocated for model operations (prompt processing, non-GPU tasks)                                 |
| `flash_attention`          | string | `"on"`  | Flash Attention mode for faster inference. Options: `"on"`, `"off"`, or `"auto"`                                        |
| `max_queue_size_per_model` | int    | 500     | Maximum number of requests that can wait in queue per model before rejecting new requests                               |
| `vram_multiplier`          | float  | 1.1     | Multiplier applied to model file size when estimating VRAM requirements (1.1 = 10% buffer)                              |

### Model Configuration (`models.{alias}`)

| Key                   | Type   | Default  | Description                                                                                           |
| --------------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------- |
| `model_path`          | string | Required | Absolute path to the GGUF model file on disk                                                          |
| `params.n_gpu_layers` | int    | 99       | Number of model layers to offload to GPU. Use `99` to offload all layers, `-1` for CPU-only inference |
| `params.n_ctx`        | int    | 4096     | Context window size in tokens. Larger values allow longer conversations but use more VRAM             |
| `params.embedding`    | bool   | false    | Set to `true` for embedding models to enable the `/v1/embeddings` endpoint                            |
| `params.type_k`       | string | `"f16"`  | Data type for KV cache keys. Options: `"f16"`, `"q8_0"`, `"q4_0"`. Lower precision saves VRAM         |
| `params.type_v`       | string | `"f16"`  | Data type for KV cache values. Options: `"f16"`, `"q8_0"`, `"q4_0"`. Lower precision saves VRAM       |

---

## Project Structure

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
├── monitoring/
│   ├── prometheus.yml         # Prometheus scrape config
│   └── grafana/
│       └── provisioning/      # Grafana datasources & dashboards
├── .example.config.json       # Example configuration
├── docker-compose.monitoring.yml  # Prometheus + Grafana
├── requirements.txt           # Python dependencies
├── run.py                     # Server entry point
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
