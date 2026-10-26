"""Main application entry point with proper dependency injection.

This module eliminates all global state by using the Flask application factory
pattern with dependency injection through ApplicationContext.
"""

import atexit
import logging
import os
import signal
import sys

from flask import Flask

from app_factory import create_app
from routes import register_routes


def is_flask_reloader() -> bool:
    """Check if we're in Flask debug reloader process."""
    return os.environ.get("WERKZEUG_RUN_MAIN") != "true"


def create_application() -> Flask:
    """Create Flask application with all dependencies properly injected."""
    app = create_app()

    # Get context from app config
    context = app.config["APP_CONTEXT"]
    logger = logging.getLogger(__name__)

    # Start cleanup scheduler if available
    if context.cleanup_scheduler and not is_flask_reloader():
        logger.info("Starting file cleanup scheduler...")
        context.cleanup_scheduler.start()

    # Register routes with context
    register_routes(app)

    # Setup shutdown handlers
    def shutdown_handler() -> None:
        """Clean shutdown handler."""
        if context.cleanup_scheduler and not is_flask_reloader():
            logger.info("Shutting down file cleanup scheduler...")
            context.cleanup_scheduler.stop()

    def signal_handler(sig: int, _frame: object) -> None:
        logger.info("Received signal %d, shutting down gracefully...", sig)
        shutdown_handler()
        sys.exit(0)

    # Register shutdown handlers
    atexit.register(shutdown_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return app


# Create app instance
app = create_application()


if __name__ == "__main__":
    # Get context for logging
    context = app.config["APP_CONTEXT"]
    logger = logging.getLogger(__name__)

    logging.getLogger("werkzeug").setLevel(logging.INFO)

    # Only log startup info once
    if not is_flask_reloader():
        logger.info("Starting Flask development server...")
        context.config.log_summary(logger)

    # Run Flask app
    app.run(debug=context.config.flask.debug, host=context.config.flask.host, port=context.config.flask.port)
