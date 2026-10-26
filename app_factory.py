"""Flask application factory with proper dependency injection.

This module provides the Flask application factory that creates and configures
the Flask app with all dependencies properly injected, eliminating global state.
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
from pathlib import Path

from flask import Flask

from application.config.logging_config import setup_logging
from application.config.system_config import SystemConfig
from application.container.service_container import ServiceContainer
from application.context.application_context import ApplicationContext
from application.services.progress_store import ThreadSafeProgressStore
from infrastructure.file.cleanup_scheduler import FileCleanupScheduler


def create_app(config_path: Path | None = None) -> Flask:
    """Create and configure Flask application with proper dependency injection.

    This factory function ensures no global state by creating all dependencies
    within the factory scope and injecting them into the Flask app.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured Flask application
    """
    # Load configuration
    if config_path and config_path.exists():
        app_config = SystemConfig.from_yaml(str(config_path))
    else:
        app_config = SystemConfig.from_yaml()

    # Use logging config from system config (respects config.yaml + debug mode)
    logging_config = app_config.logging_config
    if logging_config.file_path is None:
        log_dir = Path.home() / ".verbatimpapers"
        log_dir.mkdir(exist_ok=True)
        logging_config = replace(logging_config, file_path=str(log_dir / "app.log"))

    setup_logging(logging_config)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured: level=%s, file=%s", logging_config.level, logging_config.file_path)

    # Create service container
    logger.info("Initializing services...")
    service_container = ServiceContainer(app_config)

    # Create cleanup scheduler if enabled
    cleanup_scheduler = None
    if app_config.cleanup.enabled:
        from domain.interfaces import IFileManager

        file_manager = service_container.get(IFileManager)
        cleanup_scheduler = FileCleanupScheduler(
            file_manager=file_manager,
            max_file_age_seconds=int(app_config.cleanup.max_file_age_hours * 3600),
            check_interval_seconds=300,
        )

    # Bounded worker pool for background document processing: caps concurrent
    # operations at the configured limit, uploads beyond it queue up.
    background_executor = ThreadPoolExecutor(
        max_workers=app_config.performance.max_concurrent_operations,
        thread_name_prefix="verbatimpapers-worker",
    )
    # On process exit: let in-flight conversions finish, but drop queued ones
    # instead of draining the whole queue before shutdown completes.
    atexit.register(background_executor.shutdown, wait=False, cancel_futures=True)

    # Create application context
    app_context = ApplicationContext(
        config=app_config,
        service_container=service_container,
        progress_store=service_container.get(ThreadSafeProgressStore),
        background_executor=background_executor,
        cleanup_scheduler=cleanup_scheduler,
    )

    # Create Flask app
    app = Flask(__name__)

    # Configure Flask
    app.config["UPLOAD_FOLDER"] = app_config.files.upload_folder
    app.config["AUDIO_FOLDER"] = app_config.files.audio_folder
    app.config["MAX_CONTENT_LENGTH"] = app_config.files.max_file_size_mb * 1024 * 1024

    # Inject application context (no global state!)
    app.config["APP_CONTEXT"] = app_context

    # Create directories
    Path(app_config.files.upload_folder).mkdir(parents=True, exist_ok=True)
    Path(app_config.files.audio_folder).mkdir(parents=True, exist_ok=True)

    # Register error handlers
    register_error_handlers(app, app_context)

    logger.info("Flask application created successfully")
    return app


def register_error_handlers(app: Flask, context: ApplicationContext) -> None:
    """Register error handlers with access to application context."""
    logger = logging.getLogger(__name__)

    @app.errorhandler(413)
    def too_large(e: object) -> tuple[str, int]:
        max_size = context.config.files.max_file_size_mb
        return f"File is too large. Maximum file size is {max_size}MB.", 413

    @app.errorhandler(Exception)
    def handle_exception(e: Exception) -> tuple[str, int]:
        import traceback

        logger.error("Unhandled exception occurred")
        logger.error("Full traceback: %s", traceback.format_exc())
        logger.error("Exception type: %s", type(e).__name__)
        logger.error("Exception message: %s", str(e))
        return f"An error occurred: {e!s}", 500
