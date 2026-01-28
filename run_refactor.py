"""
Refactored Application Runner

This script initializes and runs the refactored Router Model application.
It handles configuration loading, status server initialization, and starting the FastAPI app.
"""

from app_refactor.core.logging_server import setup_logging
from pathlib import Path
import uvicorn
import logging
import threading
import argparse
import os
import sys
from app_refactor.core.config import load_config
from app_refactor.status_server import init_status_file, run_status_server_thread
from app_refactor.main import create_app
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Adjust path to include current directory
sys.path.append(str(Path(__file__).parent))

# Setup logging with timestamp and milliseconds format
setup_logging(log_level=logging.INFO, use_structured=False)
logger = logging.getLogger("runner")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Router Model Application")
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to config JSON file (default: config.json or CONFIG_PATH env var)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # 1. Load Configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # 2. Initialize Status Server resources
        logger.info("Initializing status resources...")
        project_root = Path(__file__).parent
        log_dir = project_root / "logs"
        init_status_file(config.dict(), log_dir)

        # 3. Start Status Server (Background Thread)
        stop_event = threading.Event()
        api_config = config.api
        status_port = 8001

        status_thread = threading.Thread(
            target=run_status_server_thread,
            args=(api_config.host, status_port, stop_event, log_dir),
            daemon=True
        )
        status_thread.start()
        logger.info(f"Status server started on port {status_port}")

        # 4. Create FastAPI App
        logger.info("Creating FastAPI application...")
        app = create_app(config)

        # 5. Run Server
        logger.info(
            f"Starting API Server at {api_config.host}:{api_config.port}")

        # We run uvicorn programmatically
        uvicorn.run(
            app,
            host=api_config.host,
            port=api_config.port,
            log_level="info",
        )

    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt")
    except Exception as e:
        logger.fatal(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if 'stop_event' in locals():
            logger.info("Stopping status server...")
            stop_event.set()

        if 'status_thread' in locals() and status_thread.is_alive():
            status_thread.join(timeout=2.0)
            logger.info("Info: Status server thread joined")


if __name__ == "__main__":
    main()
