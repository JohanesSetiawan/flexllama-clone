# RouterModelCustom API Documentation

> **Version:** 2.0.0 (Refactored)  
> **Base URL:** `http://localhost:8000`  
> **Last Updated:** January 2026

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Common Response Formats](#common-response-formats)
- [Error Handling](#error-handling)
- [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
  - [POST /v1/chat/completions](#post-v1chatcompletions)
  - [POST /v1/embeddings](#post-v1embeddings)
  - [GET /v1/models](#get-v1models)
- [Model Management](#model-management)
  - [POST /v1/models/eject](#post-v1modelseject)
  - [GET /v1/models/{model_alias}/status](#get-v1modelsmodel_aliasstatus)
  - [POST /v1/models/{model_alias}/reset](#post-v1modelsmodel_aliasreset)
  - [GET /v1/models/failed](#get-v1modelsfailed)
  - [GET /v1/models/status](#get-v1modelsstatus)
  - [GET /v1/models/status/stream](#get-v1modelsstatusstream)
- [Health & Monitoring](#health--monitoring)
  - [GET /health](#get-health)
  - [GET /ready](#get-ready)
  - [GET /live](#get-live)
  - [GET /metrics](#get-metrics)
  - [GET /metrics/stream](#get-metricsstream)
  - [GET /metrics/report](#get-metricsreport)
  - [GET /v1/telemetry/summary](#get-v1telemetrysummary)
  - [GET /v1/health/models](#get-v1healthmodels)
  - [GET /v1/queue/stats](#get-v1queuestats)
- [VRAM Monitoring](#vram-monitoring)
  - [GET /vram](#get-vram)
  - [GET /vram/summary](#get-vramsummary)
  - [GET /vram/models/{model_alias}](#get-vrammodelsmodel_alias)

---

## Overview

RouterModelCustom provides an OpenAI-compatible API layer for managing multiple LLM models running on llama.cpp servers. The router handles:

- **Dynamic model loading/unloading** based on demand
- **Priority-based request queuing** for fair resource distribution
- **VRAM management** to prevent out-of-memory errors
- **Real-time monitoring** via SSE streams and Prometheus metrics

All inference endpoints are designed to be drop-in replacements for OpenAI's API, allowing existing applications to seamlessly switch to local models.

---

## Authentication

The API supports optional Bearer token authentication.

### Enabling Authentication

1. Set `API_KEY` in your `.env` file:

```bash
API_KEY=your_secret_key_here
```

2. Include the token in requests:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your_secret_key_here"
```

### Disabling Authentication

If `API_KEY` is not set or empty, authentication is completely disabled and all requests are allowed.

### Public Endpoints

These endpoints always bypass authentication:

- `/docs` - OpenAPI documentation
- `/redoc` - ReDoc documentation
- `/openapi.json` - OpenAPI schema
- `/health` - Health check

---

## Common Response Formats

### Success Response

All successful responses return HTTP 2xx status codes with JSON bodies.

### Timestamps

All timestamps are returned in ISO 8601 format: `YYYY-MM-DDTHH:MM:SS.sssZ`

---

## Error Handling

The API uses standard HTTP status codes and returns consistent error responses:

| Status Code                 | Description                                                |
| --------------------------- | ---------------------------------------------------------- |
| `400 Bad Request`           | Invalid request body or missing required fields            |
| `404 Not Found`             | Requested model or resource does not exist                 |
| `503 Service Unavailable`   | Server is at capacity, queue is full, or insufficient VRAM |
| `504 Gateway Timeout`       | Request exceeded timeout waiting for model response        |
| `500 Internal Server Error` | Unexpected server error                                    |

### Error Response Format

```json
{
  "detail": "Human-readable error message explaining what went wrong"
}
```

### Insufficient VRAM Error (503)

When a model cannot be loaded due to insufficient GPU memory:

```json
{
  "detail": {
    "error": {
      "message": "Cannot load model: need 4500 MB VRAM, only 2048 MB available",
      "type": "insufficient_vram_error",
      "code": "vram_exhausted",
      "model": "qwen-7b",
      "required_mb": 4500,
      "available_mb": 2048,
      "loaded_models": ["llama-3b", "phi-2"]
    }
  }
}
```

---

## OpenAI-Compatible Endpoints

These endpoints follow the OpenAI API specification and can be used with any OpenAI client library.

---

### POST /v1/chat/completions

Generate chat completions from a language model. Supports both streaming and non-streaming responses.

#### Request Headers

| Header               | Type   | Required | Description                                                     |
| -------------------- | ------ | -------- | --------------------------------------------------------------- |
| `Content-Type`       | string | Yes      | Must be `application/json`                                      |
| `X-Request-Priority` | string | No       | Request priority: `high`, `normal`, or `low`. Default: `normal` |

#### Request Body

| Field         | Type         | Required | Description                                                      |
| ------------- | ------------ | -------- | ---------------------------------------------------------------- |
| `model`       | string       | Yes      | Alias of the model to use (must match a model defined in config) |
| `messages`    | array        | Yes      | Array of message objects representing the conversation           |
| `stream`      | boolean      | No       | Enable streaming response. Default: `false`                      |
| `temperature` | number       | No       | Sampling temperature (0.0 to 2.0). Default: model-dependent      |
| `max_tokens`  | integer      | No       | Maximum tokens to generate. Default: model context limit         |
| `top_p`       | number       | No       | Nucleus sampling probability. Default: 1.0                       |
| `stop`        | string/array | No       | Stop sequences to end generation                                 |

#### Message Object

| Field     | Type   | Description                           |
| --------- | ------ | ------------------------------------- |
| `role`    | string | One of: `system`, `user`, `assistant` |
| `content` | string | The message content                   |

#### Request Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Request-Priority: high" \
  -d '{
    "model": "qwen-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### Response (Non-Streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1702200000,
  "model": "qwen-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 150,
    "total_tokens": 178
  }
}
```

#### Response (Streaming)

When `stream: true`, the response is sent as Server-Sent Events (SSE):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702200000,"model":"qwen-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702200000,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":"Quantum"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1702200000,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":" computing"},"finish_reason":null}]}

data: [DONE]
```

#### Context Shift Warning

If the conversation exceeds the model's context window, the response includes a warning:

```json
{
  "choices": [...],
  "usage": {...},
  "metadata": {
    "context_shifted": true,
    "warning": "Context window exceeded. Some earlier messages were shifted out. Consider using shorter conversations."
  }
}
```

---

### POST /v1/embeddings

Generate vector embeddings for input text. The target model must have `embedding: true` set in configuration.

#### Request Body

| Field   | Type         | Required | Description                     |
| ------- | ------------ | -------- | ------------------------------- |
| `model` | string       | Yes      | Alias of the embedding model    |
| `input` | string/array | Yes      | Text or array of texts to embed |

#### Request Example

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-large",
    "input": ["Hello world", "Goodbye world"]
  }'
```

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0091, 0.0412, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.0018, -0.0072, 0.0398, ...],
      "index": 1
    }
  ],
  "model": "bge-large",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

---

### GET /v1/models

List all models available in the configuration.

#### Request Example

```bash
curl http://localhost:8000/v1/models
```

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-7b",
      "object": "model",
      "owned_by": "user",
      "n_ctx": 8192
    },
    {
      "id": "llama-3b",
      "object": "model",
      "owned_by": "user",
      "n_ctx": 4096
    },
    {
      "id": "bge-large",
      "object": "model",
      "owned_by": "user",
      "n_ctx": 512
    }
  ]
}
```

---

## Model Management

Endpoints for controlling model lifecycle: loading, unloading, and status monitoring.

---

### POST /v1/models/eject

Manually unload a model from GPU memory. Useful for freeing VRAM before loading a larger model.

#### Request Body

| Field   | Type   | Required | Description                  |
| ------- | ------ | -------- | ---------------------------- |
| `model` | string | Yes      | Alias of the model to unload |

#### Request Example

```bash
curl -X POST http://localhost:8000/v1/models/eject \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-7b"}'
```

#### Response (Success)

```json
{
  "status": "success",
  "model_ejected": "qwen-7b",
  "message": "Model 'qwen-7b' berhasil dihentikan"
}
```

#### Response (Model Not Loaded)

```json
{
  "status": "not_found",
  "model_ejected": null,
  "message": "Model 'qwen-7b' tidak sedang berjalan."
}
```

---

### GET /v1/models/{model_alias}/status

Get the current loading status of a specific model.

#### Path Parameters

| Parameter     | Type   | Description              |
| ------------- | ------ | ------------------------ |
| `model_alias` | string | The model alias to query |

#### Request Example

```bash
curl http://localhost:8000/v1/models/qwen-7b/status
```

#### Response

```json
{
  "model": "qwen-7b",
  "status": "ready",
  "port": 8085,
  "is_alive": true,
  "last_used": "2024-12-10T10:30:00.000Z",
  "vram_used_mb": 4096
}
```

#### Possible Status Values

| Status     | Description                                  |
| ---------- | -------------------------------------------- |
| `off`      | Model is not loaded                          |
| `starting` | llama-server process is being spawned        |
| `loading`  | Model weights are being loaded into VRAM     |
| `ready`    | Model is fully loaded and accepting requests |
| `stopping` | Model is being unloaded                      |
| `crashed`  | Model process crashed unexpectedly           |
| `failed`   | Model failed to load after multiple retries  |

---

### POST /v1/models/{model_alias}/reset

Clear the failure status for a model, allowing it to be loaded again. Use this after fixing configuration issues.

#### Path Parameters

| Parameter     | Type   | Description              |
| ------------- | ------ | ------------------------ |
| `model_alias` | string | The model alias to reset |

#### Request Example

```bash
curl -X POST http://localhost:8000/v1/models/qwen-7b/reset
```

#### Response (Success)

```json
{
  "status": "success",
  "model": "qwen-7b",
  "message": "Model failure status cleared. Had 3 failed attempts.",
  "previous_error": "CUDA out of memory..."
}
```

#### Response (No Failure Record)

```json
{
  "status": "not_found",
  "model": "qwen-7b",
  "message": "Model 'qwen-7b' has no failure record."
}
```

---

### GET /v1/models/failed

List all models that have failed to load.

#### Request Example

```bash
curl http://localhost:8000/v1/models/failed
```

#### Response

```json
{
  "failed_models": [
    {
      "model": "giant-70b",
      "attempts": 3,
      "error": "Insufficient VRAM: need 35000 MB, have 8192 MB"
    }
  ]
}
```

#### Response (No Failures)

```json
{
  "failed_models": [],
  "message": "No failed models"
}
```

---

### GET /v1/models/status

Get comprehensive status of all models in the system.

#### Request Example

```bash
curl http://localhost:8000/v1/models/status
```

#### Response

```json
{
  "server": {
    "status": "ready",
    "uptime_seconds": 3600
  },
  "models": {
    "qwen-7b": {
      "status": "ready",
      "port": 8085,
      "vram_used_mb": 4096,
      "last_request": "2024-12-10T10:30:00.000Z"
    },
    "llama-3b": {
      "status": "off",
      "port": null,
      "vram_used_mb": 0
    },
    "bge-large": {
      "status": "loading",
      "port": 8086,
      "vram_used_mb": 0,
      "progress": "Loading weights..."
    }
  },
  "summary": {
    "ready": 1,
    "loading": 1,
    "off": 1,
    "total": 3
  }
}
```

---

### GET /v1/models/status/stream

Real-time model status updates via Server-Sent Events (SSE). Connect once and receive push notifications for all status changes.

#### Request Example

```bash
curl -N http://localhost:8000/v1/models/status/stream
```

#### Event Types

| Event           | Description                                           |
| --------------- | ----------------------------------------------------- |
| `full_status`   | Complete status snapshot (sent on initial connection) |
| `model_update`  | Single model status changed                           |
| `server_update` | Server-level status changed                           |
| `heartbeat`     | Keep-alive signal (every 30 seconds)                  |

#### Event Stream Example

```
event: full_status
data: {"server":{"status":"ready"},"models":{"qwen-7b":{"status":"ready"}},"summary":{"ready":1}}

event: model_update
data: {"alias":"llama-3b","status":"loading","port":8086}

event: model_update
data: {"alias":"llama-3b","status":"ready","port":8086,"vram_mb":2048}

event: heartbeat
data: {"timestamp":"2024-12-10T10:31:00.000Z","server_status":"ready"}
```

#### JavaScript Client Example

```javascript
const eventSource = new EventSource("/v1/models/status/stream");

eventSource.addEventListener("full_status", (e) => {
  const data = JSON.parse(e.data);
  console.log("Initial status:", data);
});

eventSource.addEventListener("model_update", (e) => {
  const data = JSON.parse(e.data);
  console.log(`Model ${data.alias} is now ${data.status}`);
});

eventSource.addEventListener("heartbeat", (e) => {
  console.log("Connection alive");
});

eventSource.onerror = (e) => {
  console.error("SSE connection error");
};
```

---

## Health & Monitoring

Endpoints for monitoring server health, performance metrics, and request telemetry.

---

### GET /health

Comprehensive health check for the router and its dependencies.

#### Request Example

```bash
curl http://localhost:8000/health
```

#### Response (Healthy)

```json
{
  "status": "ok",
  "checks": {
    "gpu": {
      "status": "ok",
      "vram_used_gb": "4.25"
    },
    "manager": {
      "status": "ok",
      "active_models": 2
    },
    "http_client": {
      "status": "ok"
    }
  }
}
```

#### Response (Degraded - HTTP 503)

```json
{
  "status": "degraded",
  "checks": {
    "gpu": {
      "status": "error",
      "error": "NVML device not found"
    },
    "manager": {
      "status": "ok",
      "active_models": 1
    },
    "http_client": {
      "status": "ok"
    }
  }
}
```

---

### GET /ready

Kubernetes readiness probe. Returns 200 OK when the application is ready to serve traffic.

#### Request Example

```bash
curl http://localhost:8000/ready
```

#### Response (Ready)

```json
{
  "status": "ready"
}
```

#### Response (Not Ready)

```json
{
  "status": "not_ready"
}
```

---

### GET /live

Kubernetes liveness probe. Returns 200 OK as long as the server process is running.

#### Request Example

```bash
curl http://localhost:8000/live
```

#### Response

```json
{
  "status": "alive"
}
```

---

### GET /metrics

Prometheus-compatible metrics endpoint. Designed for scraping by Prometheus server.

#### Request Example

```bash
curl http://localhost:8000/metrics
```

#### Response (Prometheus Text Format)

```
# HELP router_requests_total Total number of requests processed
# TYPE router_requests_total counter
router_requests_total{model="qwen-7b",endpoint="/v1/chat/completions",status="success"} 1234
router_requests_total{model="qwen-7b",endpoint="/v1/chat/completions",status="error"} 12

# HELP router_request_duration_seconds Request latency in seconds
# TYPE router_request_duration_seconds histogram
router_request_duration_seconds_bucket{model="qwen-7b",le="0.5"} 100
router_request_duration_seconds_bucket{model="qwen-7b",le="1.0"} 500
router_request_duration_seconds_bucket{model="qwen-7b",le="+Inf"} 1234

# HELP router_model_vram_bytes Current VRAM usage per model
# TYPE router_model_vram_bytes gauge
router_model_vram_bytes{model="qwen-7b"} 4294967296

# HELP router_queue_depth Current number of requests waiting in queue
# TYPE router_queue_depth gauge
router_queue_depth{model="qwen-7b"} 5

# HELP router_models_loaded Number of models currently loaded
# TYPE router_models_loaded gauge
router_models_loaded 2
```

---

### GET /metrics/stream

Real-time metrics updates via Server-Sent Events.

#### Request Example

```bash
curl -N http://localhost:8000/metrics/stream
```

#### Event Types

| Event       | Description                             |
| ----------- | --------------------------------------- |
| `metrics`   | Metrics snapshot (every 2 seconds)      |
| `heartbeat` | Keep-alive signal (every 30 seconds)    |
| `error`     | Error occurred while collecting metrics |

#### Event Stream Example

```
event: metrics
data: {"timestamp":"2024-12-10T10:30:00.000Z","gpu":{"used_gb":4.25,"free_gb":3.75},"models":{"qwen-7b":{"requests":100,"latency_avg":0.5}}}

event: heartbeat
data: {"timestamp":"2024-12-10T10:30:30.000Z"}
```

---

### GET /metrics/report

Get a comprehensive 5-minute aggregated metrics report. Ideal for dashboards and alerting.

#### Request Example

```bash
curl http://localhost:8000/metrics/report
```

#### Response

```json
{
  "generated_at": "2024-12-10T10:30:00.000Z",
  "period_minutes": 5,
  "server": {
    "status": "healthy",
    "uptime_seconds": 7200
  },
  "gpu": {
    "total_gb": 8.0,
    "used_gb": 4.25,
    "free_gb": 3.75,
    "usage_percentage": 53.1
  },
  "models": {
    "qwen-7b": {
      "status": "ready",
      "requests": {
        "total": 500,
        "success": 495,
        "errors": 5,
        "success_rate": 99.0
      },
      "latency": {
        "avg_ms": 450,
        "min_ms": 120,
        "max_ms": 2500,
        "p95_ms": 1200
      },
      "queue": {
        "current_depth": 3,
        "total_processed": 495,
        "total_rejected": 2
      },
      "tokens": {
        "total_generated": 125000,
        "avg_per_request": 250
      }
    }
  },
  "totals": {
    "total_requests": 500,
    "total_success": 495,
    "total_errors": 5,
    "total_tokens": 125000
  }
}
```

---

### GET /v1/telemetry/summary

Get telemetry summary with request statistics.

#### Request Example

```bash
curl http://localhost:8000/v1/telemetry/summary
```

#### Response

```json
{
  "total_requests": 10000,
  "success_count": 9850,
  "error_count": 150,
  "avg_latency_ms": 420,
  "p95_latency_ms": 1100,
  "tokens_generated": 2500000,
  "requests_by_model": {
    "qwen-7b": 6000,
    "llama-3b": 3500,
    "bge-large": 500
  },
  "errors_by_type": {
    "timeout": 100,
    "vram_exhausted": 30,
    "model_crashed": 20
  }
}
```

---

### GET /v1/health/models

Get health status for all currently loaded models based on periodic health checks.

#### Request Example

```bash
curl http://localhost:8000/v1/health/models
```

#### Response

```json
{
  "qwen-7b": {
    "status": "healthy",
    "uptime_percentage": "99.5%",
    "consecutive_failures": 0,
    "avg_response_time_ms": "15.2",
    "last_check": "2024-12-10T10:30:00.000Z"
  },
  "llama-3b": {
    "status": "degraded",
    "uptime_percentage": "85.0%",
    "consecutive_failures": 2,
    "avg_response_time_ms": "45.8",
    "last_check": "2024-12-10T10:30:00.000Z"
  }
}
```

---

### GET /v1/queue/stats

Get detailed statistics for all model request queues.

#### Request Example

```bash
curl http://localhost:8000/v1/queue/stats
```

#### Response

```json
{
  "summary": {
    "total_queued": 8,
    "total_processing": 2,
    "total_processed": 10000,
    "total_rejected": 15
  },
  "per_model": {
    "qwen-7b": {
      "queue_length": 5,
      "total_requests": 6000,
      "total_processed": 5995,
      "total_rejected": 10,
      "current_processing": 1,
      "processing": true
    },
    "llama-3b": {
      "queue_length": 3,
      "total_requests": 4000,
      "total_processed": 4005,
      "total_rejected": 5,
      "current_processing": 1,
      "processing": true
    }
  }
}
```

---

## VRAM Monitoring

Endpoints for monitoring GPU memory usage.

---

### GET /vram

Get comprehensive VRAM usage report with per-model breakdown.

#### Request Example

```bash
curl http://localhost:8000/vram
```

#### Response

```json
{
  "status": "healthy",
  "gpu_info": {
    "device_index": 0,
    "total_mb": 8192,
    "total_gb": 8.0,
    "used_mb": 4352,
    "used_gb": 4.25,
    "free_mb": 3840,
    "free_gb": 3.75,
    "usage_percentage": 53.1
  },
  "models": [
    {
      "model_alias": "qwen-7b",
      "status": "loaded",
      "port": 8085,
      "vram_used_mb": 4096,
      "vram_used_gb": 4.0,
      "vram_percentage": 50.0,
      "load_duration_sec": 12.5
    }
  ],
  "loaded_models_count": 1,
  "can_load_more": true,
  "estimated_free_for_new_model_mb": 3640
}
```

---

### GET /vram/summary

Get a simplified VRAM summary, optimized for dashboard widgets.

#### Request Example

```bash
curl http://localhost:8000/vram/summary
```

#### Response

```json
{
  "status": "healthy",
  "total_gb": 8.0,
  "used_gb": 4.25,
  "free_gb": 3.75,
  "usage_percentage": 53.1,
  "loaded_models": 1,
  "can_load_more": true,
  "models": [
    {
      "alias": "qwen-7b",
      "vram_gb": 4.0,
      "percentage": 50.0
    }
  ]
}
```

---

### GET /vram/models/{model_alias}

Get detailed VRAM usage for a specific model, including historical snapshots.

#### Path Parameters

| Parameter     | Type   | Description              |
| ------------- | ------ | ------------------------ |
| `model_alias` | string | The model alias to query |

#### Request Example

```bash
curl http://localhost:8000/vram/models/qwen-7b
```

#### Response

```json
{
  "model_alias": "qwen-7b",
  "port": 8085,
  "status": "loaded",
  "vram_usage": {
    "current_mb": 4096.0,
    "current_gb": 4.0,
    "average_mb": 4050.25,
    "percentage_of_total": 50.0
  },
  "load_info": {
    "start_time": "2024-12-10T10:00:00.000Z",
    "end_time": "2024-12-10T10:00:12.500Z",
    "duration_sec": 12.5,
    "initial_free_vram_mb": 7680.0
  },
  "snapshots": [
    {
      "timestamp": "2024-12-10T10:00:15.000Z",
      "vram_used_mb": 4096.0,
      "status": "loaded"
    },
    {
      "timestamp": "2024-12-10T10:05:15.000Z",
      "vram_used_mb": 4100.0,
      "status": "loaded"
    }
  ],
  "current_gpu_state": {
    "total_mb": 8192.0,
    "used_mb": 4352.0,
    "free_mb": 3840.0
  }
}
```

---

## Rate Limiting & Best Practices

### Request Priority

Use the `X-Request-Priority` header to influence queue ordering:

| Priority | Use Case                             |
| -------- | ------------------------------------ |
| `high`   | Real-time chat, user-facing requests |
| `normal` | Default priority for most requests   |
| `low`    | Batch processing, background jobs    |

### Handling Backpressure

When the queue is full, the API returns HTTP 503. Clients should implement exponential backoff:

```javascript
async function requestWithRetry(payload, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      if (response.status === 503) {
        const delay = Math.pow(2, i) * 1000; // 1s, 2s, 4s
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }

      return response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
    }
  }
}
```

### Streaming Recommendations

For streaming responses, ensure your client properly handles Server-Sent Events and closes connections gracefully on user navigation or cancellation.

---

## Changelog

### v1.0.0 (December 2025)

- Initial release
- OpenAI-compatible chat completions and embeddings
- Priority queue system
- VRAM management
- Prometheus metrics integration
- SSE streaming for status updates
