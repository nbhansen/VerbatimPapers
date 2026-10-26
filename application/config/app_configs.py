# application/config/app_configs.py
"""Application-level configuration classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlaskConfig:
    """Flask application configuration."""

    debug: bool = False  # True enables the Werkzeug debugger (remote code execution if exposed)
    host: str = "127.0.0.1"  # Secure default: localhost only
    port: int = 5000


@dataclass(frozen=True)
class AdminConfig:
    """Gating for the /admin/* endpoints.

    Disabled by default: the endpoints expose filesystem details and can delete
    generated audio, which is unsafe when the server binds to a non-localhost host.
    """

    enabled: bool = False
    token: str | None = None
