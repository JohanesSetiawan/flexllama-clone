"""
Telemetry Collection Module

This module provides a telemetry collection system for tracking request performance
and generating summary statistics per model.

Components:
    - RequestMetrics: Dataclass for storing metrics of a single request
    - TelemetryCollector: Collector and aggregator for telemetry data

Metrics tracked per request:
    - request_id: Unique identifier for tracking
    - model_alias: Model used for the request
    - endpoint: API endpoint called
    - start_time/end_time: Timestamps for duration calculation
    - status_code: HTTP response status code
    - error: Error message if any
    - queue_time: Time spent waiting in queue
    - processing_time: Time spent processing by model
    - tokens_generated: Number of tokens generated

Summary Statistics:
    - Total requests, success rate, error rate
    - Response time stats (avg, min, max, p50, p95)
    - Per-model breakdown with detailed metrics

Usage:
    telemetry = TelemetryCollector(window_size=1000)
    
    # Record metrics
    await telemetry.record_request(RequestMetrics(
        request_id="req-123",
        model_alias="qwen3-8b",
        endpoint="/v1/chat/completions",
        start_time=time.time(),
        ...
    ))
    
    # Get summary
    summary = telemetry.get_summary()
"""

import asyncio
import statistics
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """
    Metrics for a single request.

    Attributes:
        request_id: Unique identifier for the request
        model_alias: Model that handled the request
        endpoint: API endpoint that was called
        start_time: Request start timestamp
        end_time: Request end timestamp
        status_code: HTTP response status code
        error: Error message if request failed
        queue_time: Time spent waiting in queue (seconds)
        processing_time: Time spent processing by model (seconds)
        tokens_generated: Number of tokens generated in response
    """
    request_id: str
    model_alias: str
    endpoint: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    queue_time: float = 0.0
    processing_time: float = 0.0
    tokens_generated: int = 0

    @property
    def total_time(self) -> float:
        """Calculate total request time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_successful(self) -> bool:
        """Check if request was successful."""
        return self.error is None and (
            self.status_code is None or self.status_code < 400
        )


@dataclass
class ModelStats:
    """
    Aggregated statistics for a single model.

    Attributes:
        total_requests: Total number of requests
        total_errors: Total number of failed requests
        total_tokens: Total tokens generated
        response_times: Recent response times for percentile calculation
    """
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))


class TelemetryService:
    """
    Collect and aggregate telemetry data for request monitoring.

    This collector maintains a sliding window of recent requests and
    aggregated per-model statistics for generating summaries.

    Attributes:
        window_size: Number of recent requests to keep for statistics
        recent_requests: Deque of recent RequestMetrics
        model_stats: Dictionary of per-model statistics
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize the telemetry collector.

        Args:
            window_size: Number of recent requests to keep (default: 1000)
        """
        self.window_size = window_size
        self.recent_requests: deque[RequestMetrics] = deque(maxlen=window_size)
        self.lock = asyncio.Lock()
        self.model_stats: Dict[str, ModelStats] = {}

    async def record_request(self, metrics: RequestMetrics) -> None:
        """
        Record request metrics.

        Args:
            metrics: RequestMetrics instance to record
        """
        async with self.lock:
            self.recent_requests.append(metrics)
            self._update_model_stats(metrics)

    def _update_model_stats(self, metrics: RequestMetrics) -> None:
        """Update aggregated model statistics."""
        model_alias = metrics.model_alias

        if model_alias not in self.model_stats:
            self.model_stats[model_alias] = ModelStats()

        stats = self.model_stats[model_alias]
        stats.total_requests += 1

        if metrics.error:
            stats.total_errors += 1

        if metrics.tokens_generated:
            stats.total_tokens += metrics.tokens_generated

        if metrics.total_time > 0:
            stats.response_times.append(metrics.total_time)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary containing:
            - total_requests: Total request count
            - successful: Successful request count
            - failed: Failed request count
            - success_rate: Success percentage
            - response_time_stats: Timing statistics
            - per_model: Per-model breakdown
        """
        if not self.recent_requests:
            return {"message": "No requests recorded yet"}

        # Calculate overall stats
        total_requests = len(self.recent_requests)
        successful = sum(1 for r in self.recent_requests if r.is_successful)
        failed = total_requests - successful

        response_times = [
            r.total_time for r in self.recent_requests if r.total_time > 0
        ]

        summary: Dict[str, Any] = {
            "total_requests": total_requests,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / total_requests * 100):.2f}%",
        }

        if response_times:
            summary["response_time_stats"] = self._calculate_time_stats(
                response_times
            )

        # Calculate per-model stats (exclude "unknown")
        summary["per_model"] = self._get_per_model_summary()

        # Add note if there are "unknown" entries
        if "unknown" in self.model_stats:
            unknown_count = self.model_stats["unknown"].total_requests
            summary["_note"] = (
                f"{unknown_count} requests with unidentified model "
                f"(likely errors or monitoring endpoints)"
            )

        return summary

    def _calculate_time_stats(
        self, response_times: List[float]
    ) -> Dict[str, str]:
        """Calculate timing statistics from response times."""
        stats = {
            "avg": f"{statistics.mean(response_times):.3f}s",
            "min": f"{min(response_times):.3f}s",
            "max": f"{max(response_times):.3f}s",
            "p50": f"{statistics.median(response_times):.3f}s",
        }

        # Calculate p95 with handling for small sample sizes
        if len(response_times) >= 20:
            p95 = statistics.quantiles(response_times, n=20)[18]
            stats["p95"] = f"{p95:.3f}s"
        elif len(response_times) >= 2:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            stats["p95"] = f"{sorted_times[p95_index]:.3f}s (estimated)"
        else:
            stats["p95"] = f"{max(response_times):.3f}s (insufficient data)"

        return stats

    def _get_per_model_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for each model."""
        per_model: Dict[str, Dict[str, Any]] = {}

        for model_alias, stats in self.model_stats.items():
            # Skip "unknown" model alias (error cases)
            if model_alias == "unknown":
                continue

            response_times = list(stats.response_times)

            model_summary: Dict[str, Any] = {
                "total_requests": stats.total_requests,
                "total_errors": stats.total_errors,
                "error_rate": f"{(stats.total_errors / stats.total_requests * 100):.2f}%",
                "total_tokens": stats.total_tokens,
            }

            if response_times:
                model_summary["avg_response_time"] = (
                    f"{statistics.mean(response_times):.3f}s"
                )
                model_summary["min_response_time"] = (
                    f"{min(response_times):.3f}s"
                )
                model_summary["max_response_time"] = (
                    f"{max(response_times):.3f}s"
                )

                # Add p95 for per-model
                if len(response_times) >= 20:
                    p95 = statistics.quantiles(response_times, n=20)[18]
                    model_summary["p95_response_time"] = f"{p95:.3f}s"
                elif len(response_times) >= 2:
                    sorted_times = sorted(response_times)
                    p95_index = int(len(sorted_times) * 0.95)
                    model_summary["p95_response_time"] = (
                        f"{sorted_times[p95_index]:.3f}s (est)"
                    )
            else:
                model_summary["avg_response_time"] = "N/A"

            per_model[model_alias] = model_summary

        return per_model

    def reset(self) -> None:
        """Reset all collected telemetry data."""
        self.recent_requests.clear()
        self.model_stats.clear()
