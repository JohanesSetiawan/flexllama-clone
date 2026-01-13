"""
Status Server Module

This module provides a lightweight status server using aiohttp.
It serves status from a file, allowing monitoring even when the main application is busy or starting.
"""

import json
import asyncio
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


logger = logging.getLogger(__name__)


def init_status_file(config_data: Dict[str, Any], log_dir: Path) -> Path:
    """
    Initialize status file before FastAPI starts.
    This allows status to be read before the server is fully ready.

    Args:
        config_data: Configuration dictionary
        log_dir: Directory to store logs and status file

    Returns:
        Path to the initialized status file
    """
    status_file = log_dir / "model_status.json"

    # Ensure logs directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get model aliases from config
    model_aliases = list(config_data.get("models", {}).keys())

    # Create initial status
    status_data = {
        "server": {
            "status": "starting",
            "started_at": None,
            "updated_at": datetime.now().isoformat()
        },
        "models": {
            alias: {
                "alias": alias,
                "status": "off",
                "port": None,
                "started_at": None,
                "last_used_at": None,
                "load_progress": None,
                "error_message": None,
                "vram_used_mb": None,
                "updated_at": datetime.now().isoformat()
            }
            for alias in model_aliases
        }
    }

    # Write to file
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

    logger.info(f"Status file initialized: {status_file}")
    return status_file


def run_status_server_thread(
    host: str,
    port: int,
    stop_event: threading.Event,
    log_dir: Path
):
    """
    Run lightweight status server in a separate thread.
    This server runs faster than FastAPI and can be accessed immediately.

    Args:
        host: Host to bind to
        port: Port to bind to
        stop_event: Event to signal shutdown
        log_dir: Directory containing model_status.json
    """
    if not AIOHTTP_AVAILABLE:
        logger.warning("aiohttp not installed. Status server disabled.")
        return

    try:
        status_file = log_dir / "model_status.json"

        # Ensure logs directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

        async def get_status(request):
            """Get current status from file."""
            try:
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                    return web.json_response(data)
                return web.json_response({
                    "server": {"status": "starting"},
                    "models": {},
                    "message": "Status file not yet created"
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        async def get_model_status(request):
            """Get status for a single model."""
            model_alias = request.match_info.get('alias', '')
            try:
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                    models = data.get('models', {})
                    if model_alias in models:
                        return web.json_response(models[model_alias])
                return web.json_response(
                    {"error": f"Model '{model_alias}' not found"},
                    status=404
                )
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        async def health_check(request):
            """Simple health check."""
            return web.json_response({
                "status": "ok",
                "service": "status-server",
                "timestamp": datetime.now().isoformat()
            })

        async def sse_stream(request):
            """
            SSE stream for status updates via file polling.
            """
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['X-Accel-Buffering'] = 'no'

            await response.prepare(request)

            last_server_status = None
            last_models_status = {}
            poll_interval = 0.5
            heartbeat_interval = 30
            last_heartbeat = asyncio.get_event_loop().time()

            try:
                # Send initial full status
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        initial_data = json.load(f)

                    # Send full status
                    await response.write(
                        f"event: full_status\ndata: {json.dumps(initial_data)}\n\n".encode(
                        )
                    )

                    last_server_status = initial_data.get("server", {})
                    last_models_status = initial_data.get("models", {})

                while not stop_event.is_set():
                    await asyncio.sleep(poll_interval)
                    current_time = asyncio.get_event_loop().time()

                    # Check for file changes
                    if status_file.exists():
                        try:
                            with open(status_file, 'r') as f:
                                current_data = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            continue

                        current_server = current_data.get("server", {})
                        current_models = current_data.get("models", {})

                        # Check server status change
                        if current_server != last_server_status:
                            server_event = {
                                "server": current_server,
                                "changed_fields": [
                                    k for k in current_server
                                    if last_server_status.get(k) != current_server.get(k)
                                ] if last_server_status else ["all"]
                            }
                            await response.write(
                                f"event: server_update\ndata: {json.dumps(server_event)}\n\n".encode(
                                )
                            )
                            last_server_status = current_server.copy()
                            last_heartbeat = current_time

                        # Check individual model changes
                        changed_models = []
                        for alias, model_data in current_models.items():
                            old_model_data = last_models_status.get(alias, {})

                            if model_data != old_model_data:
                                changes = []
                                for key in ["status", "port", "load_progress", "vram_used_mb", "error_message"]:
                                    if model_data.get(key) != old_model_data.get(key):
                                        changes.append(key)

                                model_event = {
                                    "alias": alias,
                                    "previous_status": old_model_data.get("status"),
                                    "current": model_data,
                                    "changed_fields": changes
                                }
                                changed_models.append(model_event)

                        # Send model updates
                        if changed_models:
                            if len(changed_models) == 1:
                                await response.write(
                                    f"event: model_update\ndata: {json.dumps(changed_models[0])}\n\n".encode(
                                    )
                                )
                            else:
                                await response.write(
                                    f"event: models_update\ndata: {json.dumps({'models': changed_models, 'count': len(changed_models)})}\n\n".encode(
                                    )
                                )

                            last_models_status = {k: v.copy()
                                                  for k, v in current_models.items()}
                            last_heartbeat = current_time

                        # Check for removed models
                        for alias in list(last_models_status.keys()):
                            if alias not in current_models:
                                remove_event = {
                                    "alias": alias,
                                    "event": "removed"
                                }
                                await response.write(
                                    f"event: model_removed\ndata: {json.dumps(remove_event)}\n\n".encode(
                                    )
                                )
                                del last_models_status[alias]

                    # Send heartbeat
                    if current_time - last_heartbeat >= heartbeat_interval:
                        summary = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "heartbeat",
                            "server_status": last_server_status.get("status") if last_server_status else "unknown",
                            "models_summary": {}
                        }
                        for _, model in last_models_status.items():
                            m_status = model.get("status", "unknown")
                            summary["models_summary"][m_status] = summary["models_summary"].get(
                                m_status, 0) + 1

                        await response.write(
                            f"event: heartbeat\ndata: {json.dumps(summary)}\n\n".encode(
                            )
                        )
                        last_heartbeat = current_time

            except (asyncio.CancelledError, ConnectionResetError):
                pass
            except Exception as e:
                logger.error(f"SSE stream error: {e}")

            return response

        # Cors Middleware
        async def cors_middleware(app, handler):
            async def middleware_handler(request):
                if request.method == 'OPTIONS':
                    response = web.Response()
                else:
                    response = await handler(request)
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = '*'
                return response
            return middleware_handler

        app = web.Application(middlewares=[cors_middleware])
        app.router.add_get('/status', get_status)
        app.router.add_get('/status/{alias}', get_model_status)
        app.router.add_get('/health', health_check)
        app.router.add_get('/status/stream', sse_stream)

        async def run_server():
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            logger.info(f"Status server running on http://{host}:{port}")

            # Keep running until stop event
            while not stop_event.is_set():
                await asyncio.sleep(0.5)

            await runner.cleanup()
            logger.info("Status server stopped")

        # Run in new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())
        loop.close()

    except Exception as e:
        logger.error(f"Status server error: {e}")
