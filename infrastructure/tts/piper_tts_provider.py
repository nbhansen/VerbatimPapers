from collections.abc import Sequence
from contextlib import suppress

# Import logger for debugging home user issues
import logging
import os
from pathlib import Path
import ssl
import subprocess  # nosec B404
import tempfile
from typing import TYPE_CHECKING
import urllib.error
from urllib.parse import urlparse
import urllib.request

from domain.config import PiperConfig
from domain.errors import Result, tts_engine_error
from domain.interfaces import ITTSEngine
from domain.models import SynthesizedSegment
from domain.text.segmentation import split_sentences

if TYPE_CHECKING:
    from piper.config import SynthesisConfig

logger = logging.getLogger(__name__)

# Optional imports - handle gracefully at runtime
try:
    from piper.voice import PiperVoice

    PIPER_VOICE_AVAILABLE = True
    logger.debug("PiperVoice library available")
except ImportError:
    PiperVoice = None  # type: ignore[assignment, misc]
    PIPER_VOICE_AVAILABLE = False
    logger.info("PiperVoice library not available, will use CLI fallback")


class PiperTTSProvider(ITTSEngine):
    """Piper TTS Provider: local neural TTS, one segment per sentence."""

    def __init__(
        self,
        config: PiperConfig,
        repository_url: str = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0",
        project_root: str | None = None,
    ):
        """Initialize Piper TTS Provider.

        Args:
            config: Piper configuration
            repository_url: URL for downloading Piper models
            project_root: Secure project root path (for binary lookup)

        Note: Constructor never raises exceptions. Initialization errors
        are stored and returned on first call to synthesize().
        """
        self.config = config
        self.model_path = config.model_path
        self.config_path = config.config_path
        self.project_root = project_root or str(Path.cwd())  # Secure default
        self.models_dir = config.download_dir
        self.voice_instance = None
        self.repository_url = repository_url
        self._initialization_error: str | None = None  # Deferred error handling

        # Check what's available
        self._check_piper_availability()

        # Only proceed with setup if we have some form of Piper
        if not self.piper_method:
            # Will fail at generation time with helpful error
            return

        # Ensure models directory exists
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        # Auto-download model if no path specified
        if not self.model_path:
            try:
                self.model_path, self.config_path = self._ensure_model()
            except Exception as e:
                self._initialization_error = f"Model download failed: {e}"
                logger.error("Model download failed during init: %s", e)
                return

        # Make paths absolute
        if self.model_path and not Path(self.model_path).is_absolute():
            self.model_path = str(Path(self.model_path).resolve())
        if self.config_path and not Path(self.config_path).is_absolute():
            self.config_path = str(Path(self.config_path).resolve())

        # Verify files exist (defer errors instead of raising)
        if self.model_path and not Path(self.model_path).exists():
            self._initialization_error = f"Model file not found: {self.model_path}"
            logger.error("Model file not found: %s", self.model_path)
            return
        if self.config_path and not Path(self.config_path).exists():
            self._initialization_error = f"Config file not found: {self.config_path}"
            logger.error("Config file not found: %s", self.config_path)
            return

        # Initialize Python library if available
        if self.piper_method == "python_library":
            self._init_python_library()

    # === ITTSEngine Implementation ===

    def synthesize(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Synthesize text into one segment per sentence.

        Sentences are split here rather than left to espeak so that each segment
        carries the exact source text that produced its audio - that correspondence
        is what lets timing be measured instead of estimated.
        """
        logger.info("Starting Piper TTS generation for %d characters", len(text))

        # Check for deferred initialization errors
        if self._initialization_error:
            logger.error("Piper TTS initialization failed: %s", self._initialization_error)
            return Result.failure(tts_engine_error(self._initialization_error))

        if not text or text.strip() == "":
            logger.warning("Empty text provided to Piper TTS")
            return Result.failure(tts_engine_error("Empty text provided"))

        # Check if Piper is available at all
        if not hasattr(self, "piper_method") or not self.piper_method:
            logger.error("Piper TTS not available - neither python library nor command line found")
            return Result.failure(
                tts_engine_error(
                    "Piper TTS not available. Install with: pip install piper-tts or install piper command"
                )
            )

        try:
            if self.piper_method == "python_library" and self.voice_instance is not None:
                logger.debug("Using Piper Python library for TTS generation")  # type: ignore[unreachable]
                segments = self._synthesize_with_python_lib(text)
            else:
                logger.debug("Using Piper command line for TTS generation")
                segments = self._synthesize_with_command_line(text)

            if not segments:
                logger.error("TTS engine returned no audio data")
                return Result.failure(tts_engine_error("TTS engine returned no audio data"))

            logger.info("Piper TTS produced %d segment(s)", len(segments))
            return Result.success(segments)
        except subprocess.TimeoutExpired as timeout_ex:
            timeout_duration = getattr(timeout_ex, "timeout", 30)
            logger.error("Piper command timed out after %s seconds", timeout_duration)
            return Result.failure(tts_engine_error(f"Piper command timed out after {timeout_duration} seconds"))
        except Exception as e:
            logger.exception("Piper TTS generation failed: %s: %s", type(e).__name__, e)
            return Result.failure(tts_engine_error(f"Audio generation failed: {e!s}"))

    async def synthesize_async(self, text: str) -> Result[Sequence[SynthesizedSegment]]:
        """Async wrapper - Piper is local and has no native async support."""
        import asyncio

        return await asyncio.to_thread(self.synthesize, text)

    def check_ready(self) -> Result[None]:
        """Ready once a Piper runtime is found and the voice model is in place."""
        if self._initialization_error:
            return Result.failure(tts_engine_error(self._initialization_error))
        if not self.piper_method:
            return Result.failure(
                tts_engine_error(
                    "Piper TTS not available. Install with: pip install piper-tts or install piper command"
                )
            )
        if self.piper_method == "python_library" and self.voice_instance is None:
            return Result.failure(tts_engine_error(f"Could not load Piper voice model: {self.model_path}"))
        return Result.success(None)

    def _secure_download(self, url: str, destination: str, timeout: int = 30) -> None:
        """Securely download a file with SSL verification and URL validation."""
        logger.info("Starting secure download from %s to %s", url, destination)

        # Validate URL structure
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error("Invalid URL format: %s", url)
            raise ValueError(f"Invalid URL format: {url}")

        # Only allow HTTPS for security
        if parsed_url.scheme != "https":
            logger.error("Non-HTTPS URL rejected for security: %s", url)
            raise ValueError(f"Only HTTPS URLs are allowed: {url}")

        # Create SSL context with certificate verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        try:
            # Create request with timeout and proper headers
            request = urllib.request.Request(url, headers={"User-Agent": "PiperTTS/1.0"})  # nosec B310

            # Download with SSL context and timeout
            with urllib.request.urlopen(request, context=ssl_context, timeout=timeout) as response:  # nosec B310
                if response.status != 200:
                    logger.error("Download failed with HTTP %s: %s", response.status, url)
                    raise urllib.error.HTTPError(url, response.status, "Download failed", None, None)  # type: ignore[arg-type]

                # Write to destination file
                total_bytes = 0
                with Path(destination).open("wb") as f:
                    while True:
                        chunk = response.read(8192)  # Read in 8KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        total_bytes += len(chunk)

                logger.info("Download completed successfully - %d bytes written to %s", total_bytes, destination)

        except urllib.error.URLError as e:
            logger.error("Network error downloading %s: %s", url, e)
            raise Exception(f"Network error downloading {url}: {e}") from e
        except ssl.SSLError as e:
            raise Exception(f"SSL verification failed for {url}: {e}") from e
        except Exception as e:
            raise Exception(f"Download failed for {url}: {e}") from e

    # === Setup Methods ===

    def _check_piper_availability(self) -> None:
        """Check what Piper options are available."""
        self.piper_method = None
        logger.debug("Checking Piper availability...")

        # Try Python library first
        if PIPER_VOICE_AVAILABLE:
            self.piper_method = "python_library"
            logger.debug("Piper method: python_library")
            return

        # Try command line - use secure project root
        piper_cmd = str(Path(self.project_root) / "piper")
        logger.debug("Trying Piper CLI at: %s", piper_cmd)
        try:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = self.project_root + (
                (":" + env.get("LD_LIBRARY_PATH", "")) if env.get("LD_LIBRARY_PATH") else ""
            )
            result = subprocess.run([piper_cmd, "--help"], capture_output=True, text=True, timeout=5, env=env)
            if result.returncode == 0:
                self.piper_command = piper_cmd
                self.piper_method = "command_line"
                logger.debug("Piper method: command_line (%s)", piper_cmd)
                return
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug("Piper CLI not available: %s", e)

        # Nothing available
        self.piper_method = None
        logger.warning("No Piper TTS method available (neither python library nor CLI)")

    def _init_python_library(self) -> None:
        """Initialize the Python library version."""
        try:
            if PiperVoice is None:
                raise ImportError("PiperVoice not available")
            if self.model_path is None:
                raise ValueError("Piper model path is not configured")
            self.voice_instance = PiperVoice.load(self.model_path, config_path=self.config_path)
        except Exception as e:
            logger.debug("Failed to initialize Piper Python library: %s", e)
            self.voice_instance = None

    def _build_synthesis_config(self) -> "SynthesisConfig":
        """Build Piper's SynthesisConfig from our PiperConfig.

        Note the field rename: our `noise_w` is Piper's `noise_w_scale`.
        """
        from piper.config import SynthesisConfig

        return SynthesisConfig(
            speaker_id=self.config.speaker_id,
            length_scale=self.config.length_scale,
            noise_scale=self.config.noise_scale,
            noise_w_scale=self.config.noise_w,
            # Piper normalizes per *chunk*, and a chunk is one sentence - so leaving this
            # on peaks "However." to the same level as a full paragraph, and the output
            # pumps at every sentence seam. Measured on en_US-lessac-medium: peaks go
            # [1.0, 1.0] with normalization on versus [0.70, 0.52] with it off.
            normalize_audio=False,
        )

    def _synthesize_with_python_lib(self, text: str) -> list[SynthesizedSegment]:
        """Synthesize one segment per sentence using the Piper Python library.

        Each sentence is handed to Piper separately so the returned segment carries
        the exact text that produced it. Piper may split a sentence further; those
        sub-chunks are joined back into the one segment we asked for.
        """
        try:
            if self.voice_instance is None:
                raise Exception("Voice instance not initialized")

            syn_config = self._build_synthesis_config()
            sentences = split_sentences(text) or [text]

            segments: list[SynthesizedSegment] = []
            for sentence in sentences:
                if not sentence.strip():
                    continue

                chunks = list(self.voice_instance.synthesize(sentence, syn_config))
                if not chunks:
                    logger.debug("Piper produced no audio for sentence: %r", sentence[:60])
                    continue

                first = chunks[0]
                segments.append(
                    SynthesizedSegment(
                        text=sentence,
                        pcm=b"".join(chunk.audio_int16_bytes for chunk in chunks),
                        sample_rate=first.sample_rate,
                        sample_width=first.sample_width,
                        channels=first.sample_channels,
                    )
                )

            if not segments:
                raise Exception("Piper produced no audio chunks")

            return segments

        except Exception as e:
            # Loud on purpose: a silent fall-through here means every chunk pays for a
            # subprocess and a full model reload, which is easy to miss.
            logger.warning("Piper Python library generation failed, falling back to command line: %s", e)
            return self._synthesize_with_command_line(text)

    def _synthesize_with_command_line(self, text: str) -> list[SynthesizedSegment]:
        """Synthesize via the CLI, which can only return one undifferentiated segment.

        The CLI emits a finished WAV with no sentence boundaries, so timing on this
        path is necessarily coarser than the library path.
        """
        wav_bytes = self._generate_with_command_line(text)
        segment = self._segment_from_wav(text, wav_bytes)
        return [segment] if segment else []

    @staticmethod
    def _segment_from_wav(text: str, wav_bytes: bytes) -> SynthesizedSegment | None:
        """Unwrap a WAV container into a segment."""
        import io
        import wave

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                return SynthesizedSegment(
                    text=text,
                    pcm=wav_file.readframes(wav_file.getnframes()),
                    sample_rate=wav_file.getframerate(),
                    sample_width=wav_file.getsampwidth(),
                    channels=wav_file.getnchannels(),
                )
        except (wave.Error, EOFError) as e:
            logger.error("Piper CLI produced audio that could not be read as WAV: %s", e)
            return None

    def _generate_with_command_line(self, text: str) -> bytes:
        """Generate using command line."""
        temp_path = None
        timeout = max(30, len(text) // 50)  # ~1 second per 50 chars
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            # Ensure all paths are valid strings
            if not self.model_path:
                raise Exception("Model path not configured")

            cmd = [
                getattr(self, "piper_command", "piper"),
                "--model",
                self.model_path,
                "--output_file",
                temp_path,
                "--length_scale",
                str(self.config.length_scale),
                "--noise_scale",
                str(self.config.noise_scale),
                "--noise_w",
                str(self.config.noise_w),
            ]

            if self.config_path and Path(self.config_path).exists():
                cmd.extend(["--config", self.config_path])

            if self.config.speaker_id is not None:
                cmd.extend(["--speaker", str(self.config.speaker_id)])

            # Set up environment for local piper binary with libraries
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = self.project_root + (
                (":" + env.get("LD_LIBRARY_PATH", "")) if env.get("LD_LIBRARY_PATH") else ""
            )

            logger.debug("PIPER COMMAND: %s", " ".join(cmd))
            logger.debug("PIPER ENV LD_LIBRARY_PATH: %s", env.get("LD_LIBRARY_PATH"))
            logger.debug("PIPER INPUT LENGTH: %d chars", len(text))
            process = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=timeout, env=env)

            if process.returncode != 0:
                error_msg = (
                    f"Piper command failed with code {process.returncode}\n"
                    f"Command: {' '.join(cmd)}\nStderr: {process.stderr}\n"
                    f"Stdout: {process.stdout}\n"
                    f"Env LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', 'not set')}\n"
                    f"Input text length: {len(text)}\nFirst 200 chars: {text[:200]!r}"
                )
                logger.error("PIPER DEBUG: %s", error_msg)
                raise Exception(error_msg)

            if Path(temp_path).exists():
                # Verify file was created with content
                file_size = Path(temp_path).stat().st_size
                if file_size == 0:
                    raise Exception("Output file exists but contains no audio data")

                with Path(temp_path).open("rb") as f:
                    audio_data = f.read()

                if len(audio_data) > 0:
                    return audio_data
                else:
                    raise Exception("Output file exists but contains no audio data")
            else:
                raise Exception(f"Audio file was not created at {temp_path}")

        except subprocess.TimeoutExpired as e:
            raise Exception(f"Piper command timed out after {timeout} seconds") from e
        except Exception as e:
            raise Exception(f"Command line generation failed: {e}") from e
        finally:
            # Always try to clean up temp file
            if temp_path and Path(temp_path).exists():
                # Ignore cleanup errors
                with suppress(Exception):
                    Path(temp_path).unlink()

    # === Model Management ===

    def _ensure_model(self) -> tuple[str, str]:
        """Download model if needed."""
        model_name = self.config.model_name
        model_file = f"{model_name}.onnx"
        config_file = f"{model_name}.onnx.json"

        model_path = str(Path(self.models_dir) / model_file)
        config_path = str(Path(self.models_dir) / config_file)

        # Return existing model if found
        if Path(model_path).exists() and Path(config_path).exists():
            return model_path, config_path

        # Download if needed
        base_url = self.repository_url

        # Derive URL path from model name convention: {lang}_{region}-{voice}-{quality}
        # e.g. "en_US-ryan-high" -> "en/en_US/ryan/high"
        parts = model_name.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"Unrecognized Piper model name format: '{model_name}'. "
                f"Expected format: {{lang}}_{{region}}-{{voice}}-{{quality}} (e.g. en_US-ryan-high)"
            )
        lang_region, voice, quality = parts
        lang = lang_region.split("_")[0]
        model_path_segment = f"{lang}/{lang_region}/{voice}/{quality}"

        try:
            # Download model and config securely
            model_url = f"{base_url}/{model_path_segment}/{model_file}"
            config_url = f"{base_url}/{model_path_segment}/{config_file}"

            self._secure_download(model_url, model_path)
            self._secure_download(config_url, config_path)

            return model_path, config_path

        except Exception as e:
            raise Exception(f"Model download failed: {e}") from e
