"""Gemini clients retry transient errors, exercised through the real SDK against a local server.

Mocking generate_content would bypass the SDK's retry layer entirely, so these
tests serve scripted HTTP responses instead and count the requests that arrive.
"""

import base64
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from google.genai import types
import pytest

from infrastructure.gemini_http import REQUEST_TIMEOUT_MS, RETRY_OPTIONS
from infrastructure.llm.gemini_llm_provider import GeminiLLMProvider
from infrastructure.tts.gemini_tts_provider import GeminiTTSProvider

PCM = b"\x01\x00" * 240


def _audio_reply() -> dict[str, object]:
    data = base64.b64encode(PCM).decode()
    return {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": data}}]}}
        ]
    }


def _text_reply(text: str) -> dict[str, object]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _error(code: int, status: str) -> tuple[int, dict[str, object]]:
    return code, {"error": {"code": code, "message": f"scripted {status}", "status": status}}


class ScriptedGemini:
    """A local stand-in for the Gemini API that replays scripted responses in order."""

    def __init__(self) -> None:
        self.script: list[tuple[int, dict[str, object]]] = []
        self.requests = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                server.requests += 1
                status, body = server.script.pop(0)
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def http_options(self) -> types.HttpOptions:
        """Production retry policy, pointed at this server, with the backoff shrunk to milliseconds."""
        fast = RETRY_OPTIONS.model_copy(update={"initial_delay": 0.01, "max_delay": 0.02})
        return types.HttpOptions(base_url=self.base_url, retry_options=fast, timeout=5_000)


@pytest.fixture
def gemini() -> Generator[ScriptedGemini, None, None]:
    """A scripted Gemini server for the duration of one test."""
    server = ScriptedGemini()
    yield server
    server.httpd.shutdown()


def _tts(gemini: ScriptedGemini) -> GeminiTTSProvider:
    return GeminiTTSProvider(
        model_name="voice-tts", api_key="test-key", min_request_interval=0.0, http_options=gemini.http_options()
    )


class TestTTSRetries:
    """One rate limit or server hiccup must not fail a conversion."""

    def test_transient_errors_are_retried(self, gemini: ScriptedGemini) -> None:
        """A 429 and a 503 are retried; the third attempt's audio is returned."""
        gemini.script = [_error(429, "RESOURCE_EXHAUSTED"), _error(503, "UNAVAILABLE"), (200, _audio_reply())]

        result = _tts(gemini).synthesize("Hello there.")

        assert result.is_success, result.error
        assert result.value is not None and result.value[0].pcm == PCM
        assert gemini.requests == 3

    def test_client_errors_are_not_retried(self, gemini: ScriptedGemini) -> None:
        """A bad request or bad key fails at once; retrying can't fix it."""
        gemini.script = [_error(400, "INVALID_ARGUMENT")]

        result = _tts(gemini).synthesize("Hello there.")

        assert result.is_failure
        assert gemini.requests == 1

    def test_retries_are_bounded(self, gemini: ScriptedGemini) -> None:
        """A persistent rate limit fails after the configured attempts, naming the cause."""
        gemini.script = [_error(429, "RESOURCE_EXHAUSTED")] * 4

        result = _tts(gemini).synthesize("Hello there.")

        assert result.is_failure and result.error is not None
        assert "429" in (result.error.details or "")
        assert gemini.requests == 4


def test_llm_cleaning_retries_transient_errors(gemini: ScriptedGemini) -> None:
    """Text cleaning gets the same retry policy as speech."""
    gemini.script = [_error(500, "INTERNAL"), (200, _text_reply("Cleaned text."))]
    llm = GeminiLLMProvider(api_key="test-key", model_name="text-model", http_options=gemini.http_options())

    result = llm.generate_content("Clean this.")

    assert result.value == "Cleaned text."
    assert gemini.requests == 2


@pytest.mark.parametrize(
    "client",
    [
        lambda: GeminiTTSProvider(model_name="voice-tts", api_key="test-key").client,
        lambda: GeminiLLMProvider(api_key="test-key", model_name="text-model").client,
    ],
    ids=["tts", "llm"],
)
def test_production_clients_use_the_retry_policy(client: object) -> None:
    """Without explicit options, both providers build their client with retries and a timeout."""
    options = client()._api_client._http_options  # type: ignore[operator]

    assert options.retry_options == RETRY_OPTIONS
    assert options.timeout == REQUEST_TIMEOUT_MS
