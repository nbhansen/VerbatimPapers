"""Main application entry point with proper dependency injection.

This module eliminates all global state by using the Flask application factory
pattern with dependency injection through Services (application/composition.py).
"""

import atexit
import logging
import signal
import sys

from flask import Flask

from app_factory import create_app, is_serving_process
from routes import register_routes


def create_application() -> Flask:
    """Create Flask application with all dependencies properly injected."""
    app = create_app()

    # Get services from app config
    services = app.config["SERVICES"]
    logger = logging.getLogger(__name__)

    # Start cleanup scheduler if available
    if services.cleanup_scheduler and is_serving_process(services.config.flask.debug):
        logger.info("Starting file cleanup scheduler...")
        services.cleanup_scheduler.start()

    # Register routes
    register_routes(app)

    # Setup shutdown handlers
    def shutdown_handler() -> None:
        """Clean shutdown handler."""
        if services.cleanup_scheduler and is_serving_process(services.config.flask.debug):
            logger.info("Shutting down file cleanup scheduler...")
            services.cleanup_scheduler.stop()

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
    # Get services for logging
    services = app.config["SERVICES"]
    logger = logging.getLogger(__name__)

    logging.getLogger("werkzeug").setLevel(logging.INFO)

    # Only log startup info once
    if is_serving_process(services.config.flask.debug):
        logger.info("Starting Flask development server...")
        services.config.log_summary(logger)

    # Run Flask app
    app.run(debug=services.config.flask.debug, host=services.config.flask.host, port=services.config.flask.port)
