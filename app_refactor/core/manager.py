"""
Model Manager - llama-server Process Management

This module is the core of the router model that manages the lifecycle
of llama-server processes for each model.

Components:
    - RunnerProcess: Wrapper for subprocess llama-server
    - ModelManager: Manager for all active runners

RunnerProcess Responsibilities:
    - Start llama-server subprocess with appropriate configuration
    - Health check polling until model is ready
    - Graceful shutdown with escalating termination (SIGTERM -> SIGKILL)
    - Per-runner logging to file

ModelManager Responsibilities:
    - Port allocation from pool (8085-8584)
    - VRAM estimation and tracking via VRAMTracker
    - Concurrent model limiting
    - Idle timeout watchdog (auto-unload idle models)
    - Retry logic for failed loads

llama-server Command Building:
    Router automatically builds llama-server command with flags:
    - --model: Path to GGUF file
    - --host/--port: Internal binding
    - --n-gpu-layers, --ctx-size, --batch-size: From config
    - --parallel, --threads: From system config
    - -fa: Flash attention
    - --embedding: If embedding model
    - --no-context-shift: For non-SWA models (auto-detect)
    - --cache-type-k/v: KV cache quantization

Request Flow:
    1. Request enters via ModelManager.get_runner_for_request()
    2. Check if model is already running
    3. If not, estimate VRAM and check availability
    4. Start RunnerProcess and wait for ready
    5. Return runner URL for request forwarding

VRAM Management:
    - Sequential loading via load_lock (prevent race conditions)
    - Pre-load VRAM estimation: file_size * multiplier + KV cache + overhead
    - Post-load actual tracking (before/after delta)
    - Reject load if VRAM insufficient

Usage:
    manager = ModelManager(config, shutdown_event)
    runner = await manager.get_runner_for_request("qwen3-8b")
    # runner.url = "http://127.0.0.1:8085"
    
    # Stop specific model
    await manager.eject_model("qwen3-8b")
    
    # Stop all
    await manager.stop_all_runners()
"""

import gc
import os
import time
import httpx
import signal
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

from ..utils.legacy_metrics import metrics
from ..services.vram_service import VRAMService
from .errors import InsufficientVRAMError
from .config import AppConfig, ModelConfig
from ..utils.gguf_parser import get_optimal_parallel, get_model_info, GGUFModelInfo
from ..services.metrics_service import get_metrics_service


logger = logging.getLogger(__name__)


# Constants
PORT_RANGE_START = 8085
PORT_RANGE_END = 8585
GRACEFUL_SHUTDOWN_TIMEOUT = 15.0
FORCE_KILL_TIMEOUT = 5.0
HEALTH_CHECK_TIMEOUT = 0.5
DEFAULT_READY_TIMEOUT = 120
IDLE_CHECK_INTERVAL = 60
MAX_STUCK_TIME = 300


class RunnerProcess:
    """
    Wrapper for a llama-server subprocess.

    Handles starting, stopping, and health checking of a single model runner.

    Attributes:
        alias: Model alias/identifier
        config: Model configuration
        port: Assigned port number
        url: Full URL for the runner
        status: Current status (stopped, starting, loading, ready, crashed)
        process: The subprocess instance
        last_used_time: Last time the model was used
        started_time: When the model started loading
        startup_error: Error message if startup failed
    """

    # Global cache for GGUF metadata: {model_path: (parallel, model_info)}
    _gguf_cache: Dict[str, Tuple[int, Optional[GGUFModelInfo]]] = {}

    def __init__(
        self,
        alias: str,
        config: ModelConfig,
        port: int,
        llama_server_path: str,
        system_config
    ):
        """
        Initialize a runner process.

        Args:
            alias: Model alias
            config: Model configuration
            port: Port to bind to
            llama_server_path: Path to llama-server binary
            system_config: System configuration
        """
        self.alias = alias
        self.config = config
        self.port = port
        self.llama_server_path = llama_server_path
        self.system_config = system_config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.last_used_time = time.time()
        self.started_time: Optional[float] = None
        self.url = f"http://127.0.0.1:{self.port}"
        self.startup_error: Optional[str] = None
        self.status: str = "stopped"
        self.model_info: Optional[GGUFModelInfo] = None

        # Log file setup
        self.log_dir = Path("logs/runners")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{alias}_{port}.log"
        self.log_handle = None

        # Retry configuration
        self.retry_count = 0
        self.max_retries = system_config.model_load_max_retries

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        if self.process is None:
            self.status = "stopped"
            return False

        is_running = self.process.returncode is None

        if not is_running:
            self.status = "crashed"

        return is_running

    async def start(self) -> None:
        """Start the llama-server process."""
        if self.is_alive():
            logger.warning(f"[{self.alias}] Process already running")
            return

        self.started_time = time.time()
        self.status = "starting"

        # Get model info and optimal parallel setting
        parallel = self._get_optimal_parallel()

        # Build command
        command = self._build_command(parallel)

        # Memory cleanup for large models
        await self._maybe_cleanup_memory()

        # Start process
        self.startup_error = None
        self.log_handle = open(self.log_file, "w")

        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT
            )

            subprocess_start = time.time()

            # Wait for ready via health check
            health_check_start = time.time()
            await self._wait_for_ready()
            health_check_time = time.time() - health_check_start
            total_startup_time = time.time() - self.started_time

            self.last_used_time = time.time()
            self.status = "ready"

            subprocess_time = time.time() - subprocess_start
            logger.info(
                f"[{self.alias}] READY at {self.url} | "
                f"Total: {total_startup_time:.2f}s | "
                f"(subprocess: {subprocess_time:.2f}s, loading: {health_check_time:.2f}s)"
            )
        except Exception as e:
            self._close_log_handle()
            raise

    async def stop(self) -> None:
        """
        Stop the runner process gracefully with escalating termination.

        Strategy:
        1. SIGTERM (graceful) - wait up to 15 seconds
        2. SIGKILL (force) - if SIGTERM times out
        3. os.kill() as last resort
        """
        if not self.is_alive() or self.process is None:
            self.status = "stopped"
            return

        pid = self.process.pid
        logger.info(
            f"[{self.alias}] Stopping process (Port {self.port}, PID {pid})")

        try:
            # Step 1: Graceful termination with SIGTERM
            self.process.terminate()
            logger.debug(
                f"[{self.alias}] Sent SIGTERM, waiting for graceful shutdown")

            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=GRACEFUL_SHUTDOWN_TIMEOUT
                )
                logger.info(f"[{self.alias}] Stopped gracefully")
            except asyncio.TimeoutError:
                # Step 2: Escalate to SIGKILL
                await self._force_kill(pid)

        except ProcessLookupError:
            logger.info(f"[{self.alias}] Process already dead")
        except Exception as e:
            logger.error(f"[{self.alias}] Unexpected error during stop: {e}")
        finally:
            self.process = None
            self.status = "stopped"
            self._close_log_handle()

    async def _force_kill(self, pid: int) -> None:
        """Force kill the process with SIGKILL."""
        logger.warning(
            f"[{self.alias}] SIGTERM timeout. Escalating to SIGKILL"
        )

        try:
            self.process.kill()
            await asyncio.wait_for(
                self.process.wait(),
                timeout=FORCE_KILL_TIMEOUT
            )
            logger.info(f"[{self.alias}] Force killed successfully")
        except asyncio.TimeoutError:
            # Last resort: os.kill directly
            await self._os_kill(pid)
        except Exception as e:
            logger.error(f"[{self.alias}] Error during SIGKILL: {e}")

    async def _os_kill(self, pid: int) -> None:
        """Kill process using os.kill as last resort."""
        logger.warning(
            f"[{self.alias}] asyncio SIGKILL timeout. Using os.kill on PID {pid}"
        )

        try:
            os.kill(pid, signal.SIGKILL)
            await asyncio.sleep(1.0)

            # Check if process is really dead
            try:
                os.kill(pid, 0)  # Signal 0 = check if process exists
                logger.error(
                    f"[{self.alias}] Process still alive after os.kill. "
                    "May be zombie or kernel issue."
                )
            except OSError:
                logger.info(f"[{self.alias}] Killed via os.kill")
        except ProcessLookupError:
            logger.info(f"[{self.alias}] Process already dead before os.kill")
        except Exception as e:
            logger.error(f"[{self.alias}] os.kill error: {e}")

    def _close_log_handle(self) -> None:
        """Close the log file handle safely."""
        if self.log_handle and not self.log_handle.closed:
            try:
                self.log_handle.close()
            except Exception as e:
                logger.debug(f"[{self.alias}] Error closing log handle: {e}")

    def _get_optimal_parallel(self) -> int:
        """Get optimal parallel setting, using cache if available."""
        model_path = self.config.get_resolved_path()
        params = self.config.params

        if model_path in self._gguf_cache:
            parallel, model_info = self._gguf_cache[model_path]
            self.model_info = model_info
            logger.debug(f"[{self.alias}] Using cached GGUF metadata")
            return parallel

        # Calculate optimal parallel
        base_parallel = (
            params.parallel_override
            if params.parallel_override
            else self.system_config.parallel_requests
        )

        parallel, parallel_reason = get_optimal_parallel(
            model_path=model_path,
            n_ctx=params.n_ctx,
            default_parallel=base_parallel,
            min_ctx_per_slot=2048
        )

        if parallel != base_parallel:
            logger.info(f"[{self.alias}] {parallel_reason}")

        # Get and cache model info
        model_info = get_model_info(model_path)
        self._gguf_cache[model_path] = (parallel, model_info)
        self.model_info = model_info

        # Log model info
        if model_info:
            swa_status = (
                f"SWA={model_info.swa_window_size}"
                if model_info.is_swa
                else "non-SWA"
            )
            logger.info(
                f"[{self.alias}] Model: {model_info.name} | "
                f"Arch: {model_info.architecture} | {swa_status} | "
                f"Layers: {model_info.block_count} | Parallel: {parallel}"
            )

        return parallel

    def _build_command(self, parallel: int) -> List[str]:
        """Build the llama-server command line."""
        params = self.config.params
        model_path = self.config.get_resolved_path()

        command = [
            self.llama_server_path,
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--n-gpu-layers", str(params.n_gpu_layers),
            "--ctx-size", str(params.n_ctx),
            "--mlock",
            "--jinja"
        ]

        # Context shifting: disable for non-SWA models
        if self.model_info and not self.model_info.is_swa:
            command.append("--no-context-shift")
            logger.info(
                f"[{self.alias}] Non-SWA model. Context shifting DISABLED"
            )
        else:
            logger.info(
                f"[{self.alias}] SWA model. Context shifting ENABLED"
            )

        # RoPE frequency base (only if set)
        if params.rope_freq_base is not None and params.rope_freq_base > 0:
            command.extend(["--rope-freq-base", str(params.rope_freq_base)])

        # Batch size (with per-model override)
        batch_size = params.batch_override or params.n_batch
        command.extend(["--batch-size", str(batch_size)])

        # Parallel requests
        command.extend(["--parallel", str(parallel)])

        # CPU threads
        command.extend(["--threads", str(self.system_config.cpu_threads)])

        # Flash Attention
        command.extend(["-fa", self.system_config.flash_attention])

        # Memory mapping
        if not self.system_config.use_mmap:
            command.append("--no-mmap")

        # Embedding mode
        if params.embedding:
            command.append("--embedding")

        # Chat template
        if params.chat_template:
            command.extend(["--chat-template", params.chat_template])

        # Cache types
        if params.type_k and params.type_k.lower() != "none":
            command.extend(["--cache-type-k", params.type_k])

        if params.type_v and params.type_v.lower() != "none":
            command.extend(["--cache-type-v", params.type_v])

        return command

    async def _maybe_cleanup_memory(self) -> None:
        """Perform memory cleanup for large models."""
        model_path = Path(self.config.get_resolved_path())
        model_size_gb = model_path.stat().st_size / (1024 ** 3)

        if self.model_info and self.model_info.block_count > 20:
            logger.info(
                f"[{self.alias}] Large model ({model_size_gb:.1f} GB). "
                "Performing memory cleanup"
            )
            gc.collect()
            await asyncio.sleep(0.5)
            logger.info(f"[{self.alias}] Memory cleanup complete")

    async def _wait_for_ready(self, timeout: int = DEFAULT_READY_TIMEOUT) -> None:
        """Wait for the model to be ready via health check polling."""
        self.status = "loading"
        start_time = time.time()
        last_log_time = 0.0

        # Adaptive polling intervals - faster initial checks
        poll_intervals = [
            0.02, 0.02, 0.05, 0.05, 0.1, 0.1, 0.2, 0.3, 0.5, 0.5
        ]
        poll_index = 0

        async with httpx.AsyncClient() as client:
            iteration = 0
            while time.time() - start_time < timeout:
                if not self.is_alive():
                    self._handle_crash()
                    raise Exception(
                        f"Failed to start model. Error: {self.startup_error}"
                    )

                try:
                    response = await client.get(
                        f"{self.url}/health",
                        timeout=HEALTH_CHECK_TIMEOUT
                    )

                    if response.status_code == 200:
                        elapsed = time.time() - start_time
                        self.status = "ready"
                        logger.info(
                            f"[{self.alias}] READY in {elapsed:.2f}s "
                            f"(after {iteration} health checks)"
                        )
                        return

                    elif response.status_code == 503:
                        self._log_loading_progress(start_time, last_log_time)
                        last_log_time = self._maybe_update_log_time(
                            last_log_time)
                        self.status = "loading"
                    else:
                        logger.warning(
                            f"[{self.alias}] Unexpected status: {response.status_code}"
                        )

                except httpx.ConnectError:
                    self._log_waiting(start_time, last_log_time)
                    last_log_time = self._maybe_update_log_time(
                        last_log_time, 5.0)
                    self.status = "starting"

                except httpx.TimeoutException:
                    logger.debug(f"[{self.alias}] Health check timeout")

                # Adaptive sleep
                sleep_time = (
                    poll_intervals[poll_index]
                    if poll_index < len(poll_intervals)
                    else 1.0
                )
                poll_index += 1
                await asyncio.sleep(sleep_time)
                iteration += 1

        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(
            f"[{self.alias}] Failed to start after {elapsed:.1f}s "
            f"(timeout: {timeout}s)"
        )
        await self.stop()
        self.status = "crashed"
        raise TimeoutError(
            f"Runner {self.alias} failed to start within {timeout} seconds"
        )

    def _handle_crash(self) -> None:
        """Handle process crash by reading log file."""
        try:
            with open(self.log_file, "r") as f:
                self.startup_error = f.read()
        except Exception as e:
            self.startup_error = f"Process crashed, cannot read log: {e}"

        self.status = "crashed"
        logger.error(f"[{self.alias}] Crashed. Error: {self.startup_error}")

    def _log_loading_progress(
        self,
        start_time: float,
        last_log_time: float
    ) -> None:
        """Log loading progress every 3 seconds."""
        current_time = time.time()
        if current_time - last_log_time >= 3.0:
            elapsed = current_time - start_time
            logger.info(
                f"[{self.alias}] Loading... ({elapsed:.1f}s elapsed)"
            )

    def _log_waiting(
        self,
        start_time: float,
        last_log_time: float
    ) -> None:
        """Log waiting for server to start."""
        current_time = time.time()
        if current_time - last_log_time >= 5.0:
            elapsed = current_time - start_time
            logger.debug(
                f"[{self.alias}] Waiting for server... ({elapsed:.1f}s)"
            )

    def _maybe_update_log_time(
        self,
        last_log_time: float,
        interval: float = 3.0
    ) -> float:
        """Update log time if interval has passed."""
        current = time.time()
        if current - last_log_time >= interval:
            return current
        return last_log_time


class ModelManager:
    """
    Manager for all active model runners.

    Handles port allocation, VRAM management, concurrent model limiting,
    and idle timeout watchdog.

    Attributes:
        config: Application configuration
        shutdown_event: Event signaling shutdown
        active_runners: Dictionary of running models
        vram_tracker: VRAM tracking instance
        failed_models: Dictionary of failed model attempts
    """

    def __init__(self, config: AppConfig, shutdown_event: asyncio.Event):
        """
        Initialize the model manager.

        Args:
            config: Application configuration
            shutdown_event: Event signaling shutdown
        """
        self.config = config
        self.shutdown_event = shutdown_event
        self.active_runners: Dict[str, RunnerProcess] = {}
        self.port_pool = set(range(PORT_RANGE_START, PORT_RANGE_END))
        self.used_ports: set = set()
        self.lock = asyncio.Lock()
        self.gpu_devices = config.system.gpu_devices

        # Failed model tracking: {model_alias: {error: str, attempts: int}}
        self.failed_models: Dict[str, Dict[str, Any]] = {}

        # VRAM Tracker
        self.vram_service = VRAMService(
            gpu_device_index=self.gpu_devices[0],
            min_vram_required=config.system.min_vram_required
        )
        self.vram_service.start_monitoring()
        logger.info("VRAM Tracker initialized and monitoring started")

        # Start idle check watchdog
        self.check_task = asyncio.create_task(self._idle_check_watchdog())

    def _allocate_port(self) -> int:
        """Allocate an available port from the pool."""
        available_ports = self.port_pool - self.used_ports
        if not available_ports:
            raise RuntimeError("No ports available. All ports in use.")

        port = min(available_ports)
        self.used_ports.add(port)
        return port

    def _release_port(self, port: int) -> None:
        """Release a port back to the pool."""
        self.used_ports.discard(port)

    async def _idle_check_watchdog(self) -> None:
        """Background task to check and stop idle models."""
        timeout = self.config.system.idle_timeout_sec
        timeout_enabled = self.config.system.enable_idle_timeout

        if not timeout_enabled:
            logger.info(
                "[Idle Watchdog] Idle timeout DISABLED. "
                "Models will remain loaded."
            )

        try:
            while not self.shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=IDLE_CHECK_INTERVAL
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                await self._check_idle_models(timeout, timeout_enabled)

        except asyncio.CancelledError:
            logger.info("Idle check watchdog cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in idle check watchdog: {e}")

    async def _check_idle_models(
        self,
        timeout: int,
        timeout_enabled: bool
    ) -> None:
        """Check for idle or stuck models and stop them."""
        current_time = time.time()
        runners_to_stop: List[Tuple[str, int]] = []

        async with self.lock:
            for alias, runner in list(self.active_runners.items()):
                if self.shutdown_event.is_set():
                    return

                # Skip dead runners
                if not runner.is_alive():
                    continue

                # Check for stuck models
                if runner.status in ("loading", "starting"):
                    if runner.started_time:
                        stuck_time = current_time - runner.started_time
                        if stuck_time > MAX_STUCK_TIME:
                            logger.warning(
                                f"Model '{alias}' stuck at '{runner.status}' "
                                f"for {MAX_STUCK_TIME}s. Forcing stop."
                            )
                            runners_to_stop.append((alias, runner.port))
                    continue

                # Check idle timeout for ready models
                if runner.status == "ready" and timeout_enabled:
                    idle_time = current_time - runner.last_used_time
                    if idle_time > timeout:
                        logger.info(
                            f"Model '{alias}' idle for {idle_time:.0f}s "
                            f"(>{timeout}s). Stopping..."
                        )
                        runners_to_stop.append((alias, runner.port))

        # Stop runners outside lock
        for alias, port in runners_to_stop:
            await self._stop_runner(alias, port)

    async def _stop_runner(self, alias: str, port: int) -> None:
        """Stop a specific runner by alias."""
        async with self.lock:
            runner = self.active_runners.get(alias)
            if runner and runner.port == port:
                await runner.stop()
                self._release_port(port)
                del self.active_runners[alias]
                await self.vram_service.track_model_eject(alias)

    async def get_runner_for_request(
        self,
        model_alias: str
    ) -> RunnerProcess:
        """
        Get or create a runner for the specified model.

        Args:
            model_alias: Model to get runner for

        Returns:
            The running RunnerProcess instance

        Raises:
            LookupError: If model not in config
            RuntimeError: If max concurrent models reached or load fails
            InsufficientVRAMError: If not enough VRAM
        """
        if model_alias not in self.config.models:
            raise LookupError(
                f"Model '{model_alias}' not defined in config"
            )

        # Check for previously failed model
        if model_alias in self.failed_models:
            failed_info = self.failed_models[model_alias]
            if failed_info["attempts"] >= 3:
                raise RuntimeError(
                    f"Model '{model_alias}' has failed {failed_info['attempts']} times. "
                    f"Last error: {failed_info['error']}"
                )

        runner: Optional[RunnerProcess] = None

        async with self.lock:
            if self.shutdown_event.is_set():
                raise RuntimeError(
                    "Server is shutting down. Cannot start new models."
                )

            # Check existing runner
            runner = await self._check_existing_runner(model_alias)
            if runner:
                return runner

            # Check concurrent model limit
            await self._check_concurrent_limit(model_alias)

            # Create new runner
            runner = await self._create_new_runner(model_alias)

        # Start runner with retry logic (outside lock)
        return await self._start_runner_with_retry(runner, model_alias)

    async def _check_existing_runner(
        self,
        model_alias: str
    ) -> Optional[RunnerProcess]:
        """Check if an existing runner can be used."""
        if model_alias not in self.active_runners:
            return None

        runner = self.active_runners[model_alias]

        if not runner.is_alive():
            logger.warning(f"[{model_alias}] Dead runner detected")
            self._release_port(runner.port)
            del self.active_runners[model_alias]
            return None

        if runner.status in ("loading", "starting"):
            runner.last_used_time = time.time()
            logger.info(
                f"[{model_alias}] Request received while status '{runner.status}'"
            )
            return None  # Will wait for this runner

        runner.last_used_time = time.time()
        return runner

    async def _check_concurrent_limit(self, model_alias: str) -> None:
        """Check if we can start a new model."""
        active_count = sum(
            1 for r in self.active_runners.values()
            if r.is_alive() and r.status not in ("stopped", "crashed")
        )

        max_concurrent = self.config.system.max_concurrent_models

        if model_alias not in self.active_runners and active_count >= max_concurrent:
            raise RuntimeError(
                f"Maximum concurrent models ({max_concurrent}) reached"
            )

    async def _create_new_runner(
        self,
        model_alias: str
    ) -> RunnerProcess:
        """Create a new runner and allocate resources."""
        model_conf = self.config.models[model_alias]
        model_path = Path(model_conf.get_resolved_path())
        model_size_mb = model_path.stat().st_size / (1024 ** 2)

        # VRAM estimation
        estimated_vram = self._estimate_vram(model_conf, model_size_mb)

        logger.info(
            f"[{model_alias}] Preparing to load | "
            f"File: {model_size_mb:.0f} MB | "
            f"Estimated VRAM: {estimated_vram:.0f} MB"
        )

        metrics["models_loaded_total"] += 1
        new_port = self._allocate_port()

        runner = RunnerProcess(
            alias=model_alias,
            config=model_conf,
            port=new_port,
            llama_server_path=self.config.system.llama_server_path,
            system_config=self.config.system
        )

        runner.status = "starting"
        self.active_runners[model_alias] = runner

        # Track model load start (acquires load_lock)
        await self.vram_service.track_model_load_start(model_alias, new_port)

        # Check VRAM availability
        can_load, available_mb, vram_message = self.vram_service.can_load_model(
            estimated_vram_mb=estimated_vram,
            safety_buffer_mb=200
        )

        logger.info(f"[{model_alias}] VRAM check: {vram_message}")

        if not can_load:
            await self._handle_insufficient_vram(
                model_alias, runner, estimated_vram, available_mb
            )

        logger.info(
            f"[{model_alias}] VRAM check passed - proceeding with load")

        return runner

    def _estimate_vram(
        self,
        model_conf: ModelConfig,
        model_size_mb: float
    ) -> float:
        """Estimate VRAM needed for a model."""
        n_ctx = model_conf.params.n_ctx
        vram_multiplier = self.config.system.vram_multiplier

        # Base estimation
        base_vram = model_size_mb * vram_multiplier

        # KV cache estimation (~50MB per 1K context)
        kv_cache_estimate = (n_ctx / 1024) * 50

        # Total with CUDA overhead
        estimated = base_vram + kv_cache_estimate + 150

        # Ensure minimum threshold
        min_required = self.config.system.min_vram_required
        return max(estimated, min_required)

    async def _handle_insufficient_vram(
        self,
        model_alias: str,
        runner: RunnerProcess,
        estimated_vram: float,
        available_mb: float
    ) -> None:
        """Handle case when VRAM is insufficient."""
        loaded_models = [
            alias for alias, r in self.active_runners.items()
            if r.is_alive() and r.status == "ready"
        ]

        # Cleanup
        self._release_port(runner.port)
        del self.active_runners[model_alias]

        await self.vram_service.track_model_load_failed(
            model_alias,
            f"Insufficient VRAM: need {estimated_vram + 200:.0f} MB, "
            f"have {available_mb:.0f} MB"
        )

        raise InsufficientVRAMError(
            model_alias=model_alias,
            required_mb=estimated_vram + 200,
            available_mb=available_mb,
            loaded_models=loaded_models
        )

    async def _start_runner_with_retry(
        self,
        runner: RunnerProcess,
        model_alias: str
    ) -> RunnerProcess:
        """Start runner with retry logic."""
        max_retries = runner.max_retries

        for attempt in range(max_retries + 1):
            if self.shutdown_event.is_set():
                await self._abort_due_to_shutdown(model_alias, runner)
                raise RuntimeError("Server shutting down")

            try:
                if runner.status == "starting":
                    await runner.start()
                elif runner.status == "loading":
                    await runner._wait_for_ready(timeout=DEFAULT_READY_TIMEOUT)

                runner.last_used_time = time.time()

                # Clear failed status
                self.failed_models.pop(model_alias, None)

                # Track completion
                await self.vram_service.track_model_load_complete(model_alias)
                self._record_load_metrics(model_alias, runner)

                return runner

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"[{model_alias}] Attempt {attempt + 1}/{max_retries + 1} "
                    f"failed: {error_msg}"
                )

                # Check if error is retriable
                if not self._is_retriable_error(error_msg):
                    await self._handle_permanent_error(
                        model_alias, runner, error_msg, attempt
                    )
                    raise RuntimeError(
                        f"Model '{model_alias}' failed due to configuration error: "
                        f"{error_msg}"
                    )

                # Last attempt failed
                if attempt >= max_retries:
                    await self._handle_all_retries_failed(
                        model_alias, runner, error_msg, attempt
                    )
                    raise RuntimeError(
                        f"Model '{model_alias}' failed after {max_retries + 1} "
                        f"attempts. Last error: {error_msg}"
                    )

                # Retry
                await self._prepare_retry(model_alias, runner, attempt)

        raise RuntimeError(
            f"Unexpected error in retry logic for {model_alias}")

    async def _abort_due_to_shutdown(
        self,
        model_alias: str,
        runner: RunnerProcess
    ) -> None:
        """Abort load due to shutdown."""
        logger.warning(f"[{model_alias}] Aborting start due to shutdown")

        await self.vram_service.track_model_load_failed(
            model_alias, "Server is shutting down"
        )

        async with self.lock:
            if model_alias in self.active_runners:
                self._release_port(runner.port)
                del self.active_runners[model_alias]

    async def _handle_permanent_error(
        self,
        model_alias: str,
        runner: RunnerProcess,
        error_msg: str,
        attempt: int
    ) -> None:
        """Handle non-retriable error."""
        logger.error(
            f"[{model_alias}] Permanent error detected. No retry. "
            f"Error: {error_msg}"
        )

        self.failed_models[model_alias] = {
            "error": error_msg,
            "attempts": attempt + 1
        }

        await self.vram_service.track_model_load_failed(model_alias, error_msg)
        self._record_load_failed_metrics(model_alias)

        async with self.lock:
            if model_alias in self.active_runners:
                self._release_port(runner.port)
                del self.active_runners[model_alias]

    async def _handle_all_retries_failed(
        self,
        model_alias: str,
        runner: RunnerProcess,
        error_msg: str,
        attempt: int
    ) -> None:
        """Handle case when all retries are exhausted."""
        logger.error(
            f"[{model_alias}] All {attempt + 1} attempts failed. Giving up."
        )

        self.failed_models[model_alias] = {
            "error": error_msg,
            "attempts": attempt + 1
        }

        async with self.lock:
            if model_alias in self.active_runners:
                self._release_port(runner.port)
                del self.active_runners[model_alias]

    async def _prepare_retry(
        self,
        model_alias: str,
        runner: RunnerProcess,
        attempt: int
    ) -> None:
        """Prepare for retry attempt."""
        logger.info(
            f"[{model_alias}] Retry {attempt + 1}/{runner.max_retries}...")

        # Release lock before retry
        await self.vram_service.track_model_load_failed(
            model_alias, f"Retry attempt {attempt + 1} - releasing lock"
        )

        # Wait before retry with shutdown check
        for _ in range(4):
            if self.shutdown_event.is_set():
                await self._abort_due_to_shutdown(model_alias, runner)
                raise RuntimeError("Server shutting down")
            await asyncio.sleep(0.5)

        # Re-acquire lock
        await self.vram_service.track_model_load_start(model_alias, runner.port)

        # Reset runner for retry
        runner.status = "starting"
        runner.retry_count = attempt + 1

    def _record_load_metrics(
        self,
        model_alias: str,
        runner: RunnerProcess
    ) -> None:
        """Record successful load metrics to Prometheus."""
        metrics_service = get_metrics_service()
        if not metrics_service:
            return

        load_duration = (
            time.time() - runner.started_time
            if runner.started_time
            else 0
        )

        vram_bytes = 0
        if model_alias in self.vram_service.model_tracks:
            vram_mb = self.vram_service.model_tracks[model_alias].current_vram_used_mb
            vram_bytes = int(vram_mb * 1024 * 1024)

        metrics_service.record_model_load_complete(
            model_alias, load_duration, vram_bytes
        )
        metrics_service.register_model(model_alias)

        loaded_count = sum(
            1 for r in self.active_runners.values() if r.status == "ready"
        )
        metrics_service.set_models_loaded_count(loaded_count)

    def _record_load_failed_metrics(self, model_alias: str) -> None:
        """Record failed load metrics to Prometheus."""
        metrics_service = get_metrics_service()
        if metrics_service:
            metrics_service.record_model_load_failed(model_alias)

    def _is_retriable_error(self, error_msg: str) -> bool:
        """Determine if an error is retriable or permanent."""
        non_retriable_patterns = [
            "Unsupported cache type",
            "error while handling argument",
            "Model file not found",
            "Invalid model path",
            "GGML_ASSERT",
            "llama_model_load",
            "unknown argument",
            "invalid argument",
            "failed to load model",
        ]

        error_lower = error_msg.lower()
        return not any(
            pattern.lower() in error_lower
            for pattern in non_retriable_patterns
        )

    async def eject_model(self, model_alias: str) -> bool:
        """
        Eject a specific model.

        Args:
            model_alias: Model to eject

        Returns:
            True if ejected, False if model was not running
        """
        async with self.lock:
            if model_alias not in self.active_runners:
                logger.warning(f"[{model_alias}] Not currently running")
                return False

            runner = self.active_runners[model_alias]
            port = runner.port
            await runner.stop()
            self._release_port(port)
            del self.active_runners[model_alias]
            metrics["models_ejected_total"] += 1

            await self.vram_service.track_model_eject(model_alias)

            # Metrics Service
            metrics_service = get_metrics_service()
            if metrics_service:
                metrics_service.record_model_eject(model_alias)
                loaded_count = sum(
                    1 for r in self.active_runners.values()
                    if r.status == "ready"
                )
                metrics_service.set_models_loaded_count(loaded_count)

            logger.info(
                f"[{model_alias}] Ejected successfully. "
                f"Port {port} returned to pool."
            )
            return True

    async def stop_all_runners(self) -> None:
        """Stop all running models."""
        logger.info("Stopping all active runners")

        # Stop VRAM monitoring first
        await self.vram_service.stop_monitoring()

        async with self.lock:
            if not self.active_runners:
                logger.info("No active runners to stop")
                return

            model_list = list(self.active_runners.keys())
            logger.info(f"Stopping {len(model_list)} models: {model_list}")

            for alias, runner in self.active_runners.items():
                logger.info(f"Stopping runner: {alias}")
                await runner.stop()
                logger.info(f"Runner stopped: {alias}")

        self.vram_service.shutdown()
        logger.info("All runners stopped")

    async def get_model_status(self, model_alias: str) -> Dict[str, Any]:
        """
        Get status of a specific model.

        Args:
            model_alias: Model to get status for

        Returns:
            Dictionary with status information
        """
        async with self.lock:
            if model_alias not in self.config.models:
                raise LookupError(f"Model '{model_alias}' not recognized")

            if model_alias in self.active_runners:
                runner = self.active_runners[model_alias]
                if runner.status == "crashed":
                    return {"status": "crashed", "detail": runner.startup_error}
                return {"status": runner.status, "port": runner.port}

            return {"status": "stopped"}
