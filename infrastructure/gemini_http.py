"""HTTP settings shared by every Gemini client: retries and a request timeout."""

from google.genai import types

# One rate-limit or server hiccup must not fail a whole conversion: since any
# failed TTS chunk fails the document, transient errors are retried here.
# The SDK retries 408, 429 and 5xx only, with exponential backoff and jitter;
# other 4xx errors (bad key, bad request) fail at once.
RETRY_OPTIONS = types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0)

# A hung request would otherwise hold a request slot forever
REQUEST_TIMEOUT_MS = 300_000


def default_http_options() -> types.HttpOptions:
    """HTTP options for a production Gemini client."""
    return types.HttpOptions(retry_options=RETRY_OPTIONS, timeout=REQUEST_TIMEOUT_MS)
