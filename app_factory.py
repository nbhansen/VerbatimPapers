"""Flask application factory with proper dependency injection.

This module provides the Flask application factory that creates and configures
the Flask app with all dependencies properly injected, eliminating global state.
"""

import atexit
from dataclasses import replace
import logging
import os
from pathlib import Path

from flask import Flask, render_template
from werkzeug.exceptions import HTTPException

from application.composition import Services
from application.config.logging_config import setup_logging
from application.config.system_config import SystemConfig


def is_serving_process(debug: bool) -> bool:
    """Whether this process serves requests and should run background services.

    Without the reloader (debug off) there is only one process. With it, the
    parent just watches files and the child, marked WERKZEUG_RUN_MAIN, serves.
    """
    return not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"


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

    logger.info("Initializing services...")
    services = Services(app_config)

    # The reloader parent only watches files; building services there would load
    # (or download) the TTS model in a process that never converts anything.
    if is_serving_process(app_config.flask.debug):
        ready = services.warm_up()
        if ready.is_failure and ready.error is not None:
            raise RuntimeError(f"Cannot start VerbatimPapers - {ready.error.message}: {ready.error.details}")

    # On process exit: let in-flight conversions finish, but drop queued ones
    # instead of draining the whole queue before shutdown completes.
    atexit.register(services.background_executor.shutdown, wait=False, cancel_futures=True)

    # Create Flask app
    app = Flask(__name__)

    # Configure Flask
    app.config["UPLOAD_FOLDER"] = app_config.files.upload_folder
    app.config["AUDIO_FOLDER"] = app_config.files.audio_folder
    app.config["MAX_CONTENT_LENGTH"] = app_config.files.max_file_size_mb * 1024 * 1024

    # Inject services (no global state!)
    app.config["SERVICES"] = services

    # Create directories
    Path(app_config.files.upload_folder).mkdir(parents=True, exist_ok=True)
    Path(app_config.files.audio_folder).mkdir(parents=True, exist_ok=True)

    # Register error handlers
    register_error_handlers(app, app_config)

    logger.info("Flask application created successfully")
    return app


def register_error_handlers(app: Flask, config: SystemConfig) -> None:
    """Register error handlers with access to the app's configuration."""
    logger = logging.getLogger(__name__)

    @app.errorhandler(413)
    def too_large(e: object) -> tuple[str, int]:
        max_size = config.files.max_file_size_mb
        return render_template(
            "error.html", error_message=f"File is too large. Maximum file size is {max_size}MB."
        ), 413

    @app.errorhandler(Exception)
    def handle_exception(e: Exception) -> HTTPException | tuple[str, int]:
        # 404s, 405s etc. keep their own status and Werkzeug's escaped page
        if isinstance(e, HTTPException):
            return e
        # Exception text can carry paths or user input; it goes to the log only
        logger.exception("Unhandled exception occurred")
        return render_template("error.html", error_message="An unexpected error occurred. Please try again."), 500
