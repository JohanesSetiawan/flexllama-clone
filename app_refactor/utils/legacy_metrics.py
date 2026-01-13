"""
Legacy Metrics Storage Module

This module provides basic metrics storage for tracking requests and models.
Used as a legacy metrics solution before Prometheus integration.

Tracked Metrics:
    - requests_total: Total requests per endpoint
    - requests_success: Successful requests per endpoint
    - requests_failed: Failed requests per endpoint
    - request_duration_seconds: Request durations per endpoint (list for percentile calc)
    - models_loaded_total: Cumulative count of model loads
    - models_ejected_total: Cumulative count of model ejects
    - startup_time: Server startup timestamp

Note:
    For production monitoring, use the /metrics endpoint which uses
    PrometheusMetricsCollector from prometheus_metrics.py. The /metrics/legacy
    endpoint uses metrics from this module.

Usage:
    from app_refactor.utils.legacy_metrics import metrics
    
    # Increment request counter
    metrics["requests_total"]["/v1/chat/completions"] += 1
    
    # Record duration
    metrics["request_duration_seconds"]["/v1/chat/completions"].append(1.5)
"""

from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Union


# Type alias for metrics storage
MetricsDict = Dict[str, Union[
    Dict[str, int],           # Counter metrics (requests_total, etc.)
    Dict[str, List[float]],   # Duration metrics (request_duration_seconds)
    int,                      # Scalar counters (models_loaded_total)
    str                       # Timestamps (startup_time)
]]


def create_metrics_storage() -> MetricsDict:
    """
    Create a fresh metrics storage dictionary.

    Returns:
        A dictionary with all required metric keys initialized.
    """
    return {
        "requests_total": defaultdict(int),
        "requests_success": defaultdict(int),
        "requests_failed": defaultdict(int),
        "request_duration_seconds": defaultdict(list),
        "models_loaded_total": 0,
        "models_ejected_total": 0,
        "startup_time": datetime.now().isoformat()
    }


# Global metrics storage instance
# Note: This is a simple in-memory storage for legacy compatibility
metrics: MetricsDict = create_metrics_storage()


def reset_metrics() -> None:
    """
    Reset all metrics to initial state.

    Useful for testing or when metrics need to be cleared.
    """
    global metrics
    metrics = create_metrics_storage()


def increment_model_loaded() -> None:
    """Increment the model loaded counter."""
    metrics["models_loaded_total"] += 1


def increment_model_ejected() -> None:
    """Increment the model ejected counter."""
    metrics["models_ejected_total"] += 1


def record_request(
    endpoint: str,
    success: bool,
    duration_seconds: float
) -> None:
    """
    Record a request in the metrics storage.

    Args:
        endpoint: The API endpoint path
        success: Whether the request was successful
        duration_seconds: Request duration in seconds
    """
    metrics["requests_total"][endpoint] += 1

    if success:
        metrics["requests_success"][endpoint] += 1
    else:
        metrics["requests_failed"][endpoint] += 1

    # Store duration for percentile calculations
    durations = metrics["request_duration_seconds"][endpoint]
    durations.append(duration_seconds)

    # Keep only last 1000 durations per endpoint to limit memory usage
    if len(durations) > 1000:
        metrics["request_duration_seconds"][endpoint] = durations[-1000:]
