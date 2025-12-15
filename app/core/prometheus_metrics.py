"""
Prometheus Metrics Module

Provides comprehensive metrics collection using prometheus-client library.
Supports per-model detailed reporting, real-time streaming, and aggregated snapshots.

Design Principles:
- Single Responsibility: Each class handles one concern
- Open/Closed: Easy to add new metrics without modifying existing code
- Dependency Inversion: Interfaces for collectors, not concrete implementations
"""

import time
import asyncio
import logging
import pynvml
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics Definitions
# =============================================================================

# Request Metrics
REQUEST_COUNTER = Counter(
    'router_requests_total',
    'Total number of requests',
    ['model', 'status', 'endpoint']
)

REQUEST_LATENCY = Histogram(
    'router_request_duration_seconds',
    'Request latency in seconds',
    ['model', 'endpoint'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
)

ACTIVE_REQUESTS = Gauge(
    'router_active_requests',
    'Number of currently active requests',
    ['model']
)

# Queue Metrics
QUEUE_DEPTH = Gauge(
    'router_queue_depth',
    'Current queue depth per model',
    ['model']
)

QUEUE_WAIT_TIME = Histogram(
    'router_queue_wait_seconds',
    'Time spent waiting in queue',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

QUEUE_REJECTED = Counter(
    'router_queue_rejected_total',
    'Total rejected requests due to queue full',
    ['model']
)

# GPU Metrics
GPU_VRAM_USED = Gauge(
    'router_gpu_vram_used_bytes',
    'GPU VRAM currently used in bytes'
)

GPU_VRAM_FREE = Gauge(
    'router_gpu_vram_free_bytes',
    'GPU VRAM currently free in bytes'
)

GPU_VRAM_TOTAL = Gauge(
    'router_gpu_vram_total_bytes',
    'Total GPU VRAM in bytes'
)

GPU_UTILIZATION = Gauge(
    'router_gpu_utilization_percent',
    'GPU utilization percentage'
)

MODEL_VRAM = Gauge(
    'router_model_vram_bytes',
    'VRAM used by each model in bytes',
    ['model']
)

# Model Lifecycle Metrics
MODEL_STATUS = Gauge(
    'router_model_status',
    'Model status (0=stopped, 1=loading, 2=ready, 3=error)',
    ['model']
)

MODEL_LOAD_DURATION = Histogram(
    'router_model_load_duration_seconds',
    'Time to load a model',
    ['model'],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
)

MODELS_LOADED = Gauge(
    'router_models_loaded_total',
    'Total number of models currently loaded'
)

MODEL_LOADS = Counter(
    'router_model_loads_total',
    'Total model load events',
    ['model']
)

MODEL_EJECTS = Counter(
    'router_model_ejects_total',
    'Total model eject events',
    ['model']
)

# Token Metrics
TOKENS_GENERATED = Counter(
    'router_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

TOKENS_PER_SECOND = Gauge(
    'router_tokens_per_second',
    'Current tokens per second rate',
    ['model']
)

# =============================================================================
# Enhanced Metrics - Request & Traffic
# =============================================================================

REQUEST_RATE_PER_SECOND = Gauge(
    'router_request_rate_per_second',
    'Requests per second (computed)',
    ['model']
)

REQUEST_RATE_PER_MINUTE = Gauge(
    'router_request_rate_per_minute',
    'Requests per minute (computed)',
    ['model']
)

CONCURRENT_REQUESTS_PEAK = Gauge(
    'router_concurrent_requests_peak',
    'Peak concurrent requests observed',
    ['model']
)

REQUEST_STATUS_4XX = Counter(
    'router_request_4xx_total',
    'Total client error responses (4xx)',
    ['model']
)

REQUEST_STATUS_5XX = Counter(
    'router_request_5xx_total',
    'Total server error responses (5xx)',
    ['model']
)

TIMEOUT_REQUESTS = Counter(
    'router_timeout_requests_total',
    'Total requests that timed out',
    ['model']
)

# =============================================================================
# Enhanced Metrics - Latency Percentiles
# =============================================================================

LATENCY_P50 = Gauge(
    'router_latency_p50_seconds',
    'P50 (median) request latency',
    ['model']
)

LATENCY_P95 = Gauge(
    'router_latency_p95_seconds',
    'P95 request latency',
    ['model']
)

LATENCY_P99 = Gauge(
    'router_latency_p99_seconds',
    'P99 request latency',
    ['model']
)

QUEUE_WAIT_P50 = Gauge(
    'router_queue_wait_p50_seconds',
    'P50 queue wait time',
    ['model']
)

QUEUE_WAIT_P95 = Gauge(
    'router_queue_wait_p95_seconds',
    'P95 queue wait time',
    ['model']
)

QUEUE_WAIT_P99 = Gauge(
    'router_queue_wait_p99_seconds',
    'P99 queue wait time',
    ['model']
)

# =============================================================================
# Enhanced Metrics - GPU Extended
# =============================================================================

GPU_TEMPERATURE = Gauge(
    'router_gpu_temperature_celsius',
    'GPU temperature in Celsius'
)

GPU_POWER_USAGE = Gauge(
    'router_gpu_power_watts',
    'GPU power consumption in watts'
)

GPU_POWER_LIMIT = Gauge(
    'router_gpu_power_limit_watts',
    'GPU power limit in watts'
)

# =============================================================================
# Enhanced Metrics - System Resources
# =============================================================================

SYSTEM_CPU_PERCENT = Gauge(
    'router_system_cpu_percent',
    'System CPU usage percentage'
)

SYSTEM_RAM_USED_BYTES = Gauge(
    'router_system_ram_used_bytes',
    'System RAM currently used in bytes'
)

SYSTEM_RAM_TOTAL_BYTES = Gauge(
    'router_system_ram_total_bytes',
    'System RAM total in bytes'
)

SYSTEM_RAM_PERCENT = Gauge(
    'router_system_ram_percent',
    'System RAM usage percentage'
)

MODEL_SLOTS_USED = Gauge(
    'router_model_slots_used',
    'Number of model slots currently in use'
)

MODEL_SLOTS_TOTAL = Gauge(
    'router_model_slots_total',
    'Total number of model slots available'
)

# =============================================================================
# Enhanced Metrics - Token Economics
# =============================================================================

INPUT_TOKENS = Counter(
    'router_input_tokens_total',
    'Total input tokens processed',
    ['model']
)

OUTPUT_TOKENS = Counter(
    'router_output_tokens_total',
    'Total output tokens generated',
    ['model']
)

TOKEN_THROUGHPUT = Gauge(
    'router_token_throughput_per_second',
    'Token generation throughput per second',
    ['model']
)

# =============================================================================
# Enhanced Metrics - Context Window
# =============================================================================

CONTEXT_WINDOW_USAGE = Histogram(
    'router_context_window_usage_ratio',
    'Ratio of context window used (0.0-1.0)',
    ['model'],
    buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
)

CONTEXT_TOKENS_AVG = Gauge(
    'router_context_tokens_avg',
    'Average context tokens used per request',
    ['model']
)

# =============================================================================
# Enhanced Metrics - Queue Extended
# =============================================================================

QUEUE_WAIT_DISTRIBUTION = Histogram(
    'router_queue_wait_distribution_seconds',
    'Distribution of queue wait times',
    ['model'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

QUEUE_BY_PRIORITY = Gauge(
    'router_queue_by_priority',
    'Queue depth by priority level',
    ['model', 'priority']
)

# =============================================================================
# Enhanced Metrics - Reliability & SRE
# =============================================================================

UPTIME_SECONDS = Gauge(
    'router_uptime_seconds',
    'Server uptime in seconds'
)

ERROR_RATE_5M = Gauge(
    'router_error_rate_5m_percent',
    'Error rate over last 5 minutes',
    ['model']
)

RETRY_ATTEMPTS = Counter(
    'router_retry_attempts_total',
    'Total retry attempts',
    ['model']
)

RETRY_SUCCESS = Counter(
    'router_retry_success_total',
    'Successful retries after initial failure',
    ['model']
)

# SLO Tracking (targets: availability 99.5%, P95 latency < 10s)
SLO_LATENCY_VIOLATIONS = Counter(
    'router_slo_latency_violations_total',
    'Requests violating P95 latency SLO (>10s)',
    ['model']
)

SLO_AVAILABILITY_RATIO = Gauge(
    'router_slo_availability_ratio',
    'Current availability ratio (0.0-1.0)'
)

SLO_ERROR_BUDGET_REMAINING = Gauge(
    'router_slo_error_budget_remaining_percent',
    'Remaining error budget percentage (target: 99.5% availability)'
)

# Server Info
SERVER_INFO = Info(
    'router_server',
    'Router server information'
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RequestRecord:
    """Record of a single request for aggregation."""
    timestamp: float
    model: str
    endpoint: str
    duration_seconds: float
    status: str
    tokens: int = 0
    queue_wait_seconds: float = 0.0


@dataclass
class ModelMetricsSnapshot:
    """Snapshot of metrics for a single model."""
    model_alias: str
    status: str
    vram_mb: float
    queue_depth: int
    active_requests: int
    requests_total: int
    requests_success: int
    requests_failed: int
    avg_latency_ms: float
    p95_latency_ms: float
    tokens_total: int
    tokens_per_second: float


@dataclass
class AggregatedReport:
    """5-minute aggregated metrics report."""
    timestamp: str
    window_start: str
    window_end: str
    server_uptime_seconds: float
    gpu: Dict[str, Any]
    models: List[ModelMetricsSnapshot]
    totals: Dict[str, Any]


# =============================================================================
# Metrics Collector
# =============================================================================

class PrometheusMetricsCollector:
    """
    Central metrics collector that bridges application events to Prometheus metrics.

    Responsibilities:
    - Collect and record metrics from various sources
    - Maintain rolling window for aggregation
    - Provide real-time and aggregated snapshots
    """

    def __init__(self, gpu_device_index: int = 0, max_concurrent_models: int = 3):
        self.gpu_device_index = gpu_device_index
        self.gpu_handle = None
        self.start_time = time.time()

        # Rolling window for 5-minute aggregation (store last 10 minutes)
        self.request_history: deque[RequestRecord] = deque(maxlen=10000)
        self.lock = asyncio.Lock()

        # Per-model active request tracking
        self.active_requests_count: Dict[str, int] = {}

        # Token rate tracking (sliding window)
        self.token_windows: Dict[str, deque] = {}
        self.token_window_size = 60  # 60 second window for tokens/sec

        # Initialize GPU
        self._init_gpu()

        # List of registered models for initialization
        self.registered_models: List[str] = []

        # === Enhanced metrics tracking ===
        # Peak concurrent requests per model
        self.peak_concurrent: Dict[str, int] = {}

        # Latency history for percentile calculations (last 5 min)
        self.latency_history: Dict[str, deque] = {}
        self.queue_wait_history: Dict[str, deque] = {}

        # Request timestamps for rate calculation
        self.request_timestamps: Dict[str, deque] = {}

        # Context token tracking
        self.context_usage_sum: Dict[str, float] = {}
        self.context_usage_count: Dict[str, int] = {}

        # SLO tracking
        self.total_requests = 0
        self.error_requests = 0
        self.slo_latency_threshold = 10.0  # 10 seconds for P95

        # Model slots
        self.max_concurrent_models = max_concurrent_models
        MODEL_SLOTS_TOTAL.set(max_concurrent_models)

        # Set server info
        SERVER_INFO.info({
            'version': '1.0.0',
            'start_time': datetime.now().isoformat()
        })

    def register_model(self, model_alias: str):
        """Register a model and initialize its metrics to 0."""
        if model_alias not in self.registered_models:
            self.registered_models.append(model_alias)
            # Initialize gauges with 0
            ACTIVE_REQUESTS.labels(model=model_alias).set(0)
            QUEUE_DEPTH.labels(model=model_alias).set(0)
            self.active_requests_count[model_alias] = 0
            logger.debug(f"[Metrics] Registered model: {model_alias}")

    def _init_gpu(self):
        """Initialize GPU handle for metrics collection."""
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_device_index)
            logger.info(f"[Metrics] GPU initialized for metrics collection")
        except Exception as e:
            logger.warning(f"[Metrics] Failed to initialize GPU metrics: {e}")
            self.gpu_handle = None

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        if not self.gpu_handle:
            return {
                "vram_used_bytes": 0,
                "vram_free_bytes": 0,
                "vram_total_bytes": 0,
                "utilization_percent": 0
            }

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

            # Try to get GPU utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0

            return {
                "vram_used_bytes": mem_info.used,
                "vram_free_bytes": mem_info.free,
                "vram_total_bytes": mem_info.total,
                "utilization_percent": gpu_util
            }
        except Exception as e:
            logger.debug(f"[Metrics] Error getting GPU metrics: {e}")
            return {
                "vram_used_bytes": 0,
                "vram_free_bytes": 0,
                "vram_total_bytes": 0,
                "utilization_percent": 0
            }

    def update_gpu_metrics(self):
        """Update GPU-related Prometheus metrics."""
        gpu = self._get_gpu_metrics()
        GPU_VRAM_USED.set(gpu["vram_used_bytes"])
        GPU_VRAM_FREE.set(gpu["vram_free_bytes"])
        GPU_VRAM_TOTAL.set(gpu["vram_total_bytes"])
        GPU_UTILIZATION.set(gpu["utilization_percent"])

    # =========================================================================
    # Enhanced Metrics - GPU Extended
    # =========================================================================

    def update_gpu_extended_metrics(self):
        """Update extended GPU metrics (temperature, power)."""
        if not self.gpu_handle:
            return

        try:
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            GPU_TEMPERATURE.set(temp)

            # Power usage
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
            GPU_POWER_USAGE.set(power_mw / 1000.0)  # Convert to watts

            # Power limit
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(
                self.gpu_handle)
            GPU_POWER_LIMIT.set(power_limit_mw / 1000.0)  # Convert to watts

        except Exception as e:
            logger.debug(f"[Metrics] Error getting GPU extended metrics: {e}")

    # =========================================================================
    # Enhanced Metrics - System Resources
    # =========================================================================

    def update_system_metrics(self):
        """Update system CPU and RAM metrics."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_PERCENT.set(cpu_percent)

            # RAM
            mem = psutil.virtual_memory()
            SYSTEM_RAM_USED_BYTES.set(mem.used)
            SYSTEM_RAM_TOTAL_BYTES.set(mem.total)
            SYSTEM_RAM_PERCENT.set(mem.percent)

        except Exception as e:
            logger.debug(f"[Metrics] Error getting system metrics: {e}")

    def update_uptime(self):
        """Update server uptime metric."""
        uptime = time.time() - self.start_time
        UPTIME_SECONDS.set(uptime)

    # =========================================================================
    # Enhanced Metrics - Percentile Calculations
    # =========================================================================

    def _compute_percentile(self, data: List[float], percentile: float) -> float:
        """Compute percentile from a sorted list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    def update_percentile_metrics(self, model: str):
        """Compute and update P50/P95/P99 metrics for a model."""
        # Latency percentiles
        if model in self.latency_history and self.latency_history[model]:
            latencies = list(self.latency_history[model])
            LATENCY_P50.labels(model=model).set(
                self._compute_percentile(latencies, 50))
            LATENCY_P95.labels(model=model).set(
                self._compute_percentile(latencies, 95))
            LATENCY_P99.labels(model=model).set(
                self._compute_percentile(latencies, 99))

        # Queue wait percentiles
        if model in self.queue_wait_history and self.queue_wait_history[model]:
            waits = list(self.queue_wait_history[model])
            QUEUE_WAIT_P50.labels(model=model).set(
                self._compute_percentile(waits, 50))
            QUEUE_WAIT_P95.labels(model=model).set(
                self._compute_percentile(waits, 95))
            QUEUE_WAIT_P99.labels(model=model).set(
                self._compute_percentile(waits, 99))

    # =========================================================================
    # Enhanced Metrics - Request Rate Calculations
    # =========================================================================

    def update_request_rates(self, model: str):
        """Update requests per second/minute for a model."""
        current_time = time.time()

        if model not in self.request_timestamps:
            REQUEST_RATE_PER_SECOND.labels(model=model).set(0)
            REQUEST_RATE_PER_MINUTE.labels(model=model).set(0)
            return

        timestamps = self.request_timestamps[model]

        # Requests in last second
        one_sec_ago = current_time - 1
        rps = sum(1 for ts in timestamps if ts >= one_sec_ago)
        REQUEST_RATE_PER_SECOND.labels(model=model).set(rps)

        # Requests in last minute
        one_min_ago = current_time - 60
        rpm = sum(1 for ts in timestamps if ts >= one_min_ago)
        REQUEST_RATE_PER_MINUTE.labels(model=model).set(rpm)

    # =========================================================================
    # Enhanced Metrics - Status Code Tracking
    # =========================================================================

    def record_status_code(self, model: str, status_code: int):
        """Record HTTP status code breakdown."""
        if 400 <= status_code < 500:
            REQUEST_STATUS_4XX.labels(model=model).inc()
        elif status_code >= 500:
            REQUEST_STATUS_5XX.labels(model=model).inc()

    def record_timeout(self, model: str):
        """Record a timeout event."""
        TIMEOUT_REQUESTS.labels(model=model).inc()

    # =========================================================================
    # Enhanced Metrics - Token Economics
    # =========================================================================

    def record_input_tokens(self, model: str, count: int):
        """Record input tokens processed."""
        if count > 0:
            INPUT_TOKENS.labels(model=model).inc(count)

    def record_output_tokens(self, model: str, count: int):
        """Record output tokens generated."""
        if count > 0:
            OUTPUT_TOKENS.labels(model=model).inc(count)
            # Update throughput (using existing token rate logic)
            self._update_token_rate(model, count)
            TOKEN_THROUGHPUT.labels(model=model).set(
                TOKENS_PER_SECOND.labels(model=model)._value._value
                if hasattr(TOKENS_PER_SECOND.labels(model=model), '_value') else 0
            )

    # =========================================================================
    # Enhanced Metrics - Context Window
    # =========================================================================

    def record_context_usage(self, model: str, used_tokens: int, max_tokens: int):
        """Record context window utilization."""
        if max_tokens > 0:
            ratio = used_tokens / max_tokens
            CONTEXT_WINDOW_USAGE.labels(model=model).observe(ratio)

            # Update running average
            if model not in self.context_usage_sum:
                self.context_usage_sum[model] = 0.0
                self.context_usage_count[model] = 0

            self.context_usage_sum[model] += used_tokens
            self.context_usage_count[model] += 1
            avg = self.context_usage_sum[model] / \
                self.context_usage_count[model]
            CONTEXT_TOKENS_AVG.labels(model=model).set(avg)

    # =========================================================================
    # Enhanced Metrics - SLO Tracking
    # =========================================================================

    def update_slo_metrics(self):
        """Update SLO-related metrics."""
        # Availability ratio
        if self.total_requests > 0:
            availability = 1.0 - (self.error_requests / self.total_requests)
            SLO_AVAILABILITY_RATIO.set(availability)

            # Error budget (target: 99.5% availability = 0.5% error budget)
            target_availability = 0.995
            max_errors_allowed = self.total_requests * \
                (1 - target_availability)
            if max_errors_allowed > 0:
                budget_used = self.error_requests / max_errors_allowed
                budget_remaining = max(0, 100 * (1 - budget_used))
                SLO_ERROR_BUDGET_REMAINING.set(budget_remaining)
            else:
                SLO_ERROR_BUDGET_REMAINING.set(100.0)
        else:
            SLO_AVAILABILITY_RATIO.set(1.0)
            SLO_ERROR_BUDGET_REMAINING.set(100.0)

    def record_retry_attempt(self, model: str):
        """Record a retry attempt."""
        RETRY_ATTEMPTS.labels(model=model).inc()

    def record_retry_success(self, model: str):
        """Record a successful retry."""
        RETRY_SUCCESS.labels(model=model).inc()

    # =========================================================================
    # Request Tracking
    # =========================================================================

    def record_request_start(self, model: str):
        """Record start of a request."""
        current_time = time.time()

        if model not in self.active_requests_count:
            self.active_requests_count[model] = 0
        self.active_requests_count[model] += 1
        ACTIVE_REQUESTS.labels(model=model).set(
            self.active_requests_count[model])

        # Track peak concurrent
        if model not in self.peak_concurrent:
            self.peak_concurrent[model] = 0
        if self.active_requests_count[model] > self.peak_concurrent[model]:
            self.peak_concurrent[model] = self.active_requests_count[model]
            CONCURRENT_REQUESTS_PEAK.labels(
                model=model).set(self.peak_concurrent[model])

        # Track request timestamp for rate calculation
        if model not in self.request_timestamps:
            self.request_timestamps[model] = deque(maxlen=1000)
        self.request_timestamps[model].append(current_time)

        # Update request rates
        self.update_request_rates(model)

    async def record_request_end(
        self,
        model: str,
        endpoint: str,
        duration_seconds: float,
        status: str,
        tokens: int = 0,
        queue_wait_seconds: float = 0.0,
        status_code: int = 200
    ):
        """Record end of a request with all metrics."""
        # Update active count
        if model in self.active_requests_count:
            self.active_requests_count[model] = max(
                0, self.active_requests_count[model] - 1)
            ACTIVE_REQUESTS.labels(model=model).set(
                self.active_requests_count[model])

        # Record to Prometheus
        REQUEST_COUNTER.labels(model=model, status=status,
                               endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(
            model=model, endpoint=endpoint).observe(duration_seconds)

        if queue_wait_seconds > 0:
            QUEUE_WAIT_TIME.labels(model=model).observe(queue_wait_seconds)
            QUEUE_WAIT_DISTRIBUTION.labels(
                model=model).observe(queue_wait_seconds)

        if tokens > 0:
            TOKENS_GENERATED.labels(model=model).inc(tokens)
            self._update_token_rate(model, tokens)

        # === Enhanced metrics tracking ===

        # Track latency history for percentiles
        if model not in self.latency_history:
            self.latency_history[model] = deque(maxlen=500)
        self.latency_history[model].append(duration_seconds)

        # Track queue wait history for percentiles
        if model not in self.queue_wait_history:
            self.queue_wait_history[model] = deque(maxlen=500)
        if queue_wait_seconds > 0:
            self.queue_wait_history[model].append(queue_wait_seconds)

        # Update percentile metrics
        self.update_percentile_metrics(model)

        # Track SLO violations (P95 latency > 10s)
        if duration_seconds > self.slo_latency_threshold:
            SLO_LATENCY_VIOLATIONS.labels(model=model).inc()

        # Track total/error for availability
        self.total_requests += 1
        if status == "error" or status_code >= 500:
            self.error_requests += 1

        # Record status code breakdown
        self.record_status_code(model, status_code)

        # Update SLO metrics
        self.update_slo_metrics()

        # Store in history for aggregation
        async with self.lock:
            record = RequestRecord(
                timestamp=time.time(),
                model=model,
                endpoint=endpoint,
                duration_seconds=duration_seconds,
                status=status,
                tokens=tokens,
                queue_wait_seconds=queue_wait_seconds
            )
            self.request_history.append(record)

    def _update_token_rate(self, model: str, tokens: int):
        """Update tokens per second calculation."""
        current_time = time.time()

        if model not in self.token_windows:
            self.token_windows[model] = deque()

        window = self.token_windows[model]
        window.append((current_time, tokens))

        # Remove old entries outside window
        cutoff = current_time - self.token_window_size
        while window and window[0][0] < cutoff:
            window.popleft()

        # Calculate rate
        if window:
            total_tokens = sum(t[1] for t in window)
            time_span = current_time - window[0][0]
            if time_span > 0:
                rate = total_tokens / time_span
                TOKENS_PER_SECOND.labels(model=model).set(rate)

    # =========================================================================
    # Queue Tracking
    # =========================================================================

    def update_queue_depth(self, model: str, depth: int):
        """Update queue depth for a model."""
        QUEUE_DEPTH.labels(model=model).set(depth)

    def record_queue_rejected(self, model: str):
        """Record a rejected request due to queue full."""
        QUEUE_REJECTED.labels(model=model).inc()

    # =========================================================================
    # Model Lifecycle Tracking
    # =========================================================================

    def record_model_load_start(self, model: str):
        """Record start of model loading."""
        MODEL_STATUS.labels(model=model).set(1)  # 1 = loading
        MODEL_LOADS.labels(model=model).inc()

    def record_model_load_complete(self, model: str, duration_seconds: float, vram_bytes: float):
        """Record successful model load."""
        MODEL_STATUS.labels(model=model).set(2)  # 2 = ready
        MODEL_LOAD_DURATION.labels(model=model).observe(duration_seconds)
        MODEL_VRAM.labels(model=model).set(vram_bytes)
        self._update_models_loaded_count()

    def record_model_load_failed(self, model: str):
        """Record failed model load."""
        MODEL_STATUS.labels(model=model).set(3)  # 3 = error

    def record_model_eject(self, model: str):
        """Record model ejection."""
        MODEL_STATUS.labels(model=model).set(0)  # 0 = stopped
        MODEL_EJECTS.labels(model=model).inc()
        MODEL_VRAM.labels(model=model).set(0)
        self._update_models_loaded_count()

    def update_model_status(self, model: str, status: str):
        """Update model status from string."""
        status_map = {
            "stopped": 0,
            "starting": 1,
            "loading": 1,
            "ready": 2,
            "crashed": 3,
            "error": 3
        }
        numeric_status = status_map.get(status.lower(), 0)
        MODEL_STATUS.labels(model=model).set(numeric_status)

    def _update_models_loaded_count(self):
        """Update total loaded models count."""
        # This will be called by manager to set the actual count
        pass

    def set_models_loaded_count(self, count: int):
        """Set the total number of loaded models."""
        MODELS_LOADED.set(count)

    # =========================================================================
    # Prometheus Export
    # =========================================================================

    def get_prometheus_metrics(self) -> bytes:
        """Generate Prometheus exposition format."""
        # Update all metrics before export
        self.update_gpu_metrics()
        self.update_gpu_extended_metrics()
        self.update_system_metrics()
        self.update_uptime()
        return generate_latest(REGISTRY)

    def get_content_type(self) -> str:
        """Get content type for Prometheus response."""
        return CONTENT_TYPE_LATEST

    # =========================================================================
    # Aggregated Reports
    # =========================================================================

    async def get_realtime_snapshot(self, manager=None, queue_manager=None) -> Dict[str, Any]:
        """Get real-time metrics snapshot for SSE streaming."""
        self.update_gpu_metrics()
        gpu = self._get_gpu_metrics()

        # Build models list
        models = []
        if manager:
            for alias, runner in manager.active_runners.items():
                model_data = await self._build_model_snapshot(
                    alias, runner, manager, queue_manager
                )
                models.append(model_data)

        # Calculate totals from recent history (last 60 seconds)
        totals = await self._calculate_recent_totals(60)

        return {
            "timestamp": datetime.now().isoformat(),
            "server": {
                "status": "ready",
                "uptime_seconds": round(time.time() - self.start_time, 1)
            },
            "gpu": {
                "vram_total_mb": round(gpu["vram_total_bytes"] / (1024**2), 2),
                "vram_used_mb": round(gpu["vram_used_bytes"] / (1024**2), 2),
                "vram_free_mb": round(gpu["vram_free_bytes"] / (1024**2), 2),
                "vram_used_percent": round(
                    (gpu["vram_used_bytes"] / gpu["vram_total_bytes"] * 100)
                    if gpu["vram_total_bytes"] > 0 else 0, 1
                ),
                "utilization_percent": gpu["utilization_percent"]
            },
            "models": models,
            "totals": totals
        }

    async def get_5min_report(self, manager=None, queue_manager=None) -> Dict[str, Any]:
        """Get 5-minute aggregated report."""
        self.update_gpu_metrics()
        gpu = self._get_gpu_metrics()

        now = datetime.now()
        window_start = now - timedelta(minutes=5)

        # Build per-model detailed report
        models = []
        if manager:
            for alias, runner in manager.active_runners.items():
                model_report = await self._build_model_detailed_report(
                    alias, runner, manager, queue_manager, window_start.timestamp()
                )
                models.append(model_report)

        # Calculate totals for 5 minutes
        totals = await self._calculate_recent_totals(300)

        return {
            "report_type": "5_minute_aggregation",
            "generated_at": now.isoformat(),
            "window": {
                "start": window_start.isoformat(),
                "end": now.isoformat(),
                "duration_seconds": 300
            },
            "server": {
                "status": "ready",
                "uptime_seconds": round(time.time() - self.start_time, 1)
            },
            "gpu": {
                "vram_total_mb": round(gpu["vram_total_bytes"] / (1024**2), 2),
                "vram_total_gb": round(gpu["vram_total_bytes"] / (1024**3), 2),
                "vram_used_mb": round(gpu["vram_used_bytes"] / (1024**2), 2),
                "vram_used_gb": round(gpu["vram_used_bytes"] / (1024**3), 2),
                "vram_free_mb": round(gpu["vram_free_bytes"] / (1024**2), 2),
                "vram_free_gb": round(gpu["vram_free_bytes"] / (1024**3), 2),
                "vram_used_percent": round(
                    (gpu["vram_used_bytes"] / gpu["vram_total_bytes"] * 100)
                    if gpu["vram_total_bytes"] > 0 else 0, 2
                ),
                "utilization_percent": gpu["utilization_percent"]
            },
            "models": models,
            "totals": totals
        }

    async def _build_model_snapshot(
        self,
        alias: str,
        runner,
        manager,
        queue_manager
    ) -> Dict[str, Any]:
        """Build snapshot for a single model."""
        queue_depth = 0
        if queue_manager and alias in queue_manager.queues:
            queue_depth = len(queue_manager.queues[alias].queue)

        active = self.active_requests_count.get(alias, 0)

        # Get VRAM from vram_tracker
        vram_mb = 0
        if manager and hasattr(manager, 'vram_tracker'):
            if alias in manager.vram_tracker.model_tracks:
                vram_mb = manager.vram_tracker.model_tracks[alias].current_vram_used_mb

        # Get recent stats (last 60 seconds)
        recent_stats = await self._get_model_stats(alias, 60)

        return {
            "alias": alias,
            "status": runner.status if runner else "unknown",
            "port": runner.port if runner else 0,
            "vram_mb": round(vram_mb, 2),
            "queue_depth": queue_depth,
            "active_requests": active,
            "requests_1m": recent_stats["count"],
            "errors_1m": recent_stats["errors"],
            "avg_latency_ms": round(recent_stats["avg_latency"] * 1000, 2),
            "tokens_1m": recent_stats["tokens"]
        }

    async def _build_model_detailed_report(
        self,
        alias: str,
        runner,
        manager,
        queue_manager,
        since_timestamp: float
    ) -> Dict[str, Any]:
        """Build detailed report for a single model."""
        queue_depth = 0
        queue_stats = {}
        if queue_manager and alias in queue_manager.queues:
            q = queue_manager.queues[alias]
            queue_depth = len(q.queue)
            queue_stats = q.get_stats()

        active = self.active_requests_count.get(alias, 0)

        # Get VRAM from vram_tracker
        vram_mb = 0
        if manager and hasattr(manager, 'vram_tracker'):
            if alias in manager.vram_tracker.model_tracks:
                vram_mb = manager.vram_tracker.model_tracks[alias].current_vram_used_mb

        # Get stats for the window
        stats = await self._get_model_stats(alias, 300)

        return {
            "alias": alias,
            "status": runner.status if runner else "unknown",
            "port": runner.port if runner else 0,
            "vram": {
                "used_mb": round(vram_mb, 2)
            },
            "queue": {
                "current_depth": queue_depth,
                "total_requests": queue_stats.get("total_requests", 0),
                "total_processed": queue_stats.get("total_processed", 0),
                "total_rejected": queue_stats.get("total_rejected", 0)
            },
            "active_requests": active,
            "requests": {
                "total": stats["count"],
                "success": stats["count"] - stats["errors"],
                "errors": stats["errors"],
                "error_rate_percent": round(
                    (stats["errors"] / stats["count"] *
                     100) if stats["count"] > 0 else 0, 2
                )
            },
            "latency": {
                "avg_ms": round(stats["avg_latency"] * 1000, 2),
                "min_ms": round(stats["min_latency"] * 1000, 2),
                "max_ms": round(stats["max_latency"] * 1000, 2),
                "p95_ms": round(stats["p95_latency"] * 1000, 2)
            },
            "tokens": {
                "total": stats["tokens"],
                "per_second": round(stats["tokens"] / 300 if stats["tokens"] > 0 else 0, 2)
            }
        }

    async def _get_model_stats(self, model: str, window_seconds: int) -> Dict[str, Any]:
        """Get statistics for a model within a time window."""
        async with self.lock:
            cutoff = time.time() - window_seconds
            records = [r for r in self.request_history
                       if r.model == model and r.timestamp >= cutoff]

        if not records:
            return {
                "count": 0,
                "errors": 0,
                "tokens": 0,
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "p95_latency": 0
            }

        latencies = [r.duration_seconds for r in records]
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)

        return {
            "count": len(records),
            "errors": sum(1 for r in records if r.status == "error"),
            "tokens": sum(r.tokens for r in records),
            "avg_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "p95_latency": sorted_latencies[p95_index] if sorted_latencies else 0
        }

    async def _calculate_recent_totals(self, window_seconds: int) -> Dict[str, Any]:
        """Calculate totals for recent time window."""
        async with self.lock:
            cutoff = time.time() - window_seconds
            records = [r for r in self.request_history if r.timestamp >= cutoff]

        if not records:
            return {
                "requests": 0,
                "errors": 0,
                "error_rate_percent": 0,
                "avg_latency_ms": 0,
                "tokens": 0,
                "requests_per_minute": 0
            }

        total = len(records)
        errors = sum(1 for r in records if r.status == "error")
        latencies = [r.duration_seconds for r in records]
        tokens = sum(r.tokens for r in records)

        # Calculate requests per minute
        time_span = time.time() - \
            records[0].timestamp if records else window_seconds
        rpm = (total / time_span * 60) if time_span > 0 else 0

        return {
            "requests": total,
            "errors": errors,
            "error_rate_percent": round((errors / total * 100) if total > 0 else 0, 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2) if latencies else 0,
            "tokens": tokens,
            "requests_per_minute": round(rpm, 2)
        }

    def shutdown(self):
        """Cleanup resources."""
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# =============================================================================
# Global Instance
# =============================================================================

# Singleton instance - will be initialized in main.py
prometheus_collector: Optional[PrometheusMetricsCollector] = None


def init_prometheus_collector(gpu_device_index: int = 0) -> PrometheusMetricsCollector:
    """Initialize the global Prometheus collector."""
    global prometheus_collector
    if prometheus_collector is None:
        prometheus_collector = PrometheusMetricsCollector(gpu_device_index)
        logger.info("[Metrics] Prometheus metrics collector initialized")
    return prometheus_collector


def get_prometheus_collector() -> Optional[PrometheusMetricsCollector]:
    """Get the global Prometheus collector instance."""
    return prometheus_collector
