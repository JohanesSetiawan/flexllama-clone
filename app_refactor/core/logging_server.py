"""
Structured Logging Configuration Module

This module provides logging configuration with structured (JSON) format
for easier parsing and log analysis.

Features:
    - Structured JSON format for machine-readable logs
    - Human-readable format as fallback
    - Console and file handlers
    - Suppression for noisy libraries (httpx, httpcore)

Log Fields (Structured Mode):
    - timestamp: ISO format timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - module: Python module name
    - function: Function name
    - line: Line number
    - model_alias: (optional) Related model alias
    - port: (optional) Runner port
    - status: (optional) Model status
    - exception: (optional) Exception traceback

Log Files:
    - logs/api-gateway.log: Main application logs
    - logs/runners/<alias>_<port>.log: Per-runner logs

Usage:
    from app_refactor.core.logging_server import setup_logging
    
    # Setup with structured format
    setup_logging(log_level=logging.INFO, use_structured=True)
    
    # Setup with simple format
    setup_logging(log_level=logging.DEBUG, use_structured=False)
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging.

    Outputs log records as JSON objects with consistent field names,
    making logs easy to parse with tools like jq, Elasticsearch, or Loki.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log record
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add optional extra fields if present
        for field in ("model_alias", "port", "status", "request_id"):
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """
    Simple human-readable log formatter with milliseconds.

    Format: [TIMESTAMP.mmm] LEVEL:LOGGER:MESSAGE
    """

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        """
        Override formatTime to include milliseconds.

        Returns timestamp in format: YYYY-MM-DD HH:MM:SS.mmm
        """
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S")
        # Add milliseconds
        return f"{s}.{int(record.msecs):03d}"


def setup_logging(
    log_level: int = logging.INFO,
    use_structured: bool = True,
    log_dir: Optional[str] = None
) -> None:
    """
    Configure application logging.

    Sets up console and file handlers with the specified format.
    Creates log directory if it doesn't exist.

    Args:
        log_level: Minimum log level to capture (default: INFO)
        use_structured: Use JSON format if True, simple format if False
        log_dir: Directory for log files (default: "logs")
    """
    # Select formatter based on preference
    if use_structured:
        formatter = StructuredFormatter()
    else:
        formatter = SimpleFormatter()

    # Console handler - output to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler - create directory if needed
    log_path = Path(log_dir) if log_dir else Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_path / "api-gateway.log",
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    _suppress_noisy_loggers()


def _suppress_noisy_loggers() -> None:
    """Suppress verbose logging from third-party libraries."""
    noisy_loggers = [
        "httpx",
        "httpcore",
        "uvicorn.access",
        "uvicorn.error",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_runner_logger(alias: str, port: int) -> logging.Logger:
    """
    Get a logger for a specific model runner.

    Creates a dedicated log file for the runner at logs/runners/<alias>_<port>.log

    Args:
        alias: Model alias
        port: Runner port

    Returns:
        Logger configured for the runner
    """
    logger_name = f"runner.{alias}"
    logger = logging.getLogger(logger_name)

    # Create runner logs directory
    runner_log_dir = Path("logs") / "runners"
    runner_log_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler if not already present
    log_file = runner_log_dir / f"{alias}_{port}.log"

    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(SimpleFormatter())
        logger.addHandler(handler)

    return logger
