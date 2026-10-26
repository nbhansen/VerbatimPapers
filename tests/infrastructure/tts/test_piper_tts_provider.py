"""Comprehensive infrastructure tests for PiperTTSProvider.

Tests real Piper TTS functionality without mocking core provider behavior.
These tests validate actual TTS generation, model management, and error handling.
"""

from collections.abc import Generator
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest

from domain.config.tts_config import PiperConfig
from domain.errors import ErrorCode, Result
from domain.models import SynthesizedSegment
from infrastructure.tts.piper_tts_provider import PiperTTSProvider
from infrastructure.tts.text_segmenter import TextSegmenter


def _piper_chunk(frames: int = 100, sample_rate: int = 100) -> Mock:
    """A stand-in for piper's AudioChunk: 16-bit mono at a cheap sample rate."""
    chunk = Mock()
    chunk.sample_rate = sample_rate
    chunk.sample_width = 2
    chunk.sample_channels = 1
    chunk.audio_int16_bytes = b"\x01\x00" * frames
    return chunk


def _wav_bytes(frames: int = 100, sample_rate: int = 100) -> bytes:
    """A minimal but valid mono 16-bit WAV, for stubbing the CLI's output file."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


@pytest.fixture
def temp_models_dir() -> Generator[str, None, None]:
    """Create temporary directory for test models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def basic_piper_config(temp_models_dir: str) -> PiperConfig:
    """Basic PiperConfig for testing."""
    return PiperConfig(
        model_name="en_US-lessac-medium",
        model_path=None,  # Will be auto-downloaded
        config_path=None,
        download_dir=temp_models_dir,
        length_scale=1.0,
        speaker_id=None,
    )


@pytest.fixture
def custom_piper_config(temp_models_dir: str) -> PiperConfig:
    """Custom PiperConfig with specific model paths."""
    model_path = str(Path(temp_models_dir) / "test_model.onnx")
    config_path = str(Path(temp_models_dir) / "test_model.onnx.json")

    # Create dummy model files for testing
    Path(model_path).touch()
    Path(config_path).touch()

    return PiperConfig(
        model_name="en_US-lessac-medium",
        model_path=model_path,
        config_path=config_path,
        download_dir=temp_models_dir,
        length_scale=0.8,
        speaker_id=1,
    )


class TestPiperTTSProviderInitialization:
    """Test PiperTTSProvider initialization and configuration."""

    def test_init_with_basic_config(self, basic_piper_config: PiperConfig) -> None:
        """Should initialize with basic configuration."""
        provider = PiperTTSProvider(basic_piper_config)

        assert provider.config == basic_piper_config
        assert provider.models_dir == basic_piper_config.download_dir
        assert Path(provider.models_dir).exists()

    def test_init_with_custom_paths(self, custom_piper_config: PiperConfig) -> None:
        """Should initialize with custom model paths."""
        provider = PiperTTSProvider(custom_piper_config)

        assert provider.model_path == custom_piper_config.model_path
        assert provider.config_path == custom_piper_config.config_path
        assert provider.model_path and Path(provider.model_path).is_absolute()
        assert provider.config_path and Path(provider.config_path).is_absolute()

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_init_with_missing_model_files_defers_error(self, mock_run: Mock, temp_models_dir: str) -> None:
        """Should defer error for missing model files to synthesize call."""
        # Make Piper command line available so it proceeds to model check
        mock_run.return_value = Mock(returncode=0)

        config = PiperConfig(
            model_name="en_US-lessac-medium",
            model_path="/nonexistent/model.onnx",
            config_path="/nonexistent/config.json",
            download_dir=temp_models_dir,
        )

        # Init should NOT raise - uses deferred error pattern
        provider = PiperTTSProvider(config)

        # Error should be stored for later
        assert provider._initialization_error is not None
        assert "not found" in provider._initialization_error.lower()

        # Error should be returned on first generate call
        result = provider.synthesize("Test text")
        assert result.is_failure
        assert result.error is not None
        # Error details should contain the actual error message
        error_details = result.error.details or ""
        assert "not found" in error_details.lower()

    def test_init_with_custom_project_root(self, basic_piper_config: PiperConfig) -> None:
        """Should accept custom project root for secure binary lookup."""
        custom_root = "/custom/project/root"
        provider = PiperTTSProvider(basic_piper_config, project_root=custom_root)

        assert provider.project_root == custom_root

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", True)
    def test_init_detects_python_library(self, basic_piper_config: PiperConfig) -> None:
        """Should detect Python library availability."""
        provider = PiperTTSProvider(basic_piper_config)

        assert hasattr(provider, "piper_method")
        assert provider.piper_method == "python_library"

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_init_detects_command_line(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should detect command line Piper availability."""
        mock_run.return_value = Mock(returncode=0)

        provider = PiperTTSProvider(basic_piper_config)

        assert provider.piper_method == "command_line"
        assert hasattr(provider, "piper_command")


class TestPiperTTSProviderSSMLProcessing:
    """Test SSML text processing for Piper compatibility."""

    def test_process_text_for_piper_removes_all_ssml(self, basic_piper_config: PiperConfig) -> None:
        """Should remove all SSML tags since Piper doesn't support SSML."""
        provider = PiperTTSProvider(basic_piper_config)

        ssml_text = """
        <speak>
            <break time="1s"/>
            <emphasis level="strong">Important text</emphasis>
            <prosody rate="slow" pitch="low">Slow low voice</prosody>
            Regular text.
        </speak>
        """

        result = provider._process_text_for_piper(ssml_text)

        assert "<" not in result
        assert ">" not in result
        assert "Important text" in result
        assert "Slow low voice" in result
        assert "Regular text." in result

    def test_process_text_for_piper_cleans_whitespace(self, basic_piper_config: PiperConfig) -> None:
        """Should clean up whitespace after SSML removal."""
        provider = PiperTTSProvider(basic_piper_config)

        ssml_text = "Text with   <break/>   multiple    spaces."
        result = provider._process_text_for_piper(ssml_text)

        assert result == "Text with multiple spaces."

    def test_process_text_for_piper_fixes_punctuation_spacing(self, basic_piper_config: PiperConfig) -> None:
        """Should fix spacing before punctuation after tag removal."""
        provider = PiperTTSProvider(basic_piper_config)

        ssml_text = "Hello <break/> , world <break/> !"
        result = provider._process_text_for_piper(ssml_text)

        assert result == "Hello, world!"

    def test_process_text_for_piper_preserves_plain_text(self, basic_piper_config: PiperConfig) -> None:
        """Should preserve plain text without SSML tags."""
        provider = PiperTTSProvider(basic_piper_config)

        plain_text = "This is plain text without any markup."
        result = provider._process_text_for_piper(plain_text)

        assert result == plain_text


class TestPiperTTSProviderInterfaces:
    """Test ITTSEngine interface implementation."""

    def test_supports_ssml(self, basic_piper_config: PiperConfig) -> None:
        """Should return False since Piper doesn't support SSML."""
        provider = PiperTTSProvider(basic_piper_config)

        assert provider.supports_ssml() is False


class TestPiperReadiness:
    """check_ready reports what synthesize would fail on, without synthesizing."""

    @staticmethod
    def _provider(method: str | None, voice: object = None, init_error: str | None = None) -> PiperTTSProvider:
        provider = PiperTTSProvider.__new__(PiperTTSProvider)
        provider.piper_method = method
        provider.voice_instance = voice  # type: ignore[assignment]
        provider.model_path = "/models/voice.onnx"
        provider._initialization_error = init_error
        return provider

    def test_ready_with_loaded_voice(self) -> None:
        """A loaded voice is ready."""
        assert self._provider("python_library", voice=Mock()).check_ready().is_success

    def test_ready_with_command_line(self) -> None:
        """The CLI needs no voice instance."""
        assert self._provider("command_line").check_ready().is_success

    def test_reports_deferred_init_error(self) -> None:
        """A failed model download is reported, not deferred to the first upload."""
        result = self._provider("python_library", init_error="Model download failed: offline").check_ready()
        assert result.is_failure and result.error is not None
        assert result.error.details == "Model download failed: offline"

    def test_reports_missing_runtime(self) -> None:
        """No Python library and no CLI is not ready."""
        result = self._provider(None).check_ready()
        assert result.is_failure and result.error is not None
        assert "not available" in (result.error.details or "")

    def test_reports_voice_that_failed_to_load(self) -> None:
        """The library is present but the model didn't load."""
        result = self._provider("python_library").check_ready()
        assert result.is_failure and result.error is not None
        assert "/models/voice.onnx" in (result.error.details or "")


class TestPiperTTSProviderTextValidation:
    """Test text input validation for TTS generation."""

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_synthesize_rejects_empty_text(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should reject empty text input."""
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(basic_piper_config)

        result = provider.synthesize("")

        assert result.is_failure
        assert result.error is not None
        assert "Empty text provided" in str(result.error.details)

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_synthesize_rejects_whitespace_only(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should reject whitespace-only text."""
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(basic_piper_config)

        result = provider.synthesize("   \n\t  ")

        assert result.is_failure
        assert result.error is not None
        assert "Empty text provided" in str(result.error.details)

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_synthesize_rejects_ssml_only_content(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should reject content that becomes empty after SSML removal."""
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(basic_piper_config)

        ssml_only = "<break time='2s'/><silence duration='1s'/>"
        result = provider.synthesize(ssml_only)

        assert result.is_failure
        assert result.error is not None
        assert "Text processing resulted in empty content" in str(result.error.details)


class TestPiperTTSProviderAvailabilityChecking:
    """Test Piper availability detection."""

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_synthesize_fails_when_unavailable(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should fail gracefully when Piper is not available."""
        # Mock command line check to fail
        mock_run.side_effect = FileNotFoundError("piper command not found")

        provider = PiperTTSProvider(basic_piper_config)
        result = provider.synthesize("Test text")

        assert result.is_failure
        assert result.error is not None
        assert "Piper TTS not available" in str(result.error.details)
        assert "pip install piper-tts" in str(result.error.details)


class TestPiperTTSProviderCommandLineGeneration:
    """Test command line TTS generation functionality."""

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.open")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.mkdir")
    def test_generate_with_command_line_success(
        self,
        mock_mkdir: Mock,
        mock_stat: Mock,
        mock_open: Mock,
        mock_exists: Mock,
        mock_tempfile: Mock,
        mock_run: Mock,
        custom_piper_config: PiperConfig,
    ) -> None:
        """Should successfully generate audio using command line."""
        # Setup mocks
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test_audio.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp_file

        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        mock_exists.return_value = True

        # FIX: Path.stat() needs to return object with integer st_mode
        stat_result = Mock()
        stat_result.st_size = 44100
        stat_result.st_mode = 0o100644  # S_IFREG | 0644
        mock_stat.return_value = stat_result

        # Mock file reading
        fake_audio_data = _wav_bytes(frames=100)  # real WAV, since the provider unwraps it
        mock_file_handle = Mock()
        mock_file_handle.read.return_value = fake_audio_data
        mock_open.return_value.__enter__.return_value = mock_file_handle

        # Mock initial piper availability check
        initial_run_patch = patch("subprocess.run", return_value=Mock(returncode=0))
        with initial_run_patch:
            provider = PiperTTSProvider(custom_piper_config)

        # Test generation
        result = provider.synthesize("Hello, this is a test.")

        assert result.is_success
        assert result.value is not None
        # The CLI emits a finished WAV with no sentence boundaries, so it unwraps to
        # exactly one segment carrying the whole text
        assert len(result.value) == 1
        assert result.value[0].text == "Hello, this is a test."

        # Verify command construction
        mock_run.assert_called()
        call_args = mock_run.call_args
        cmd = call_args[0][0]

        assert "piper" in cmd[0]
        assert "--model" in cmd
        assert custom_piper_config.model_path in cmd
        assert "--output_file" in cmd
        assert "--length_scale" in cmd
        assert "0.8" in cmd  # custom length scale
        assert "--speaker" in cmd
        assert "1" in cmd  # custom speaker ID
        # Voice-shaping settings must reach the CLI, not just length_scale (issue #59).
        # Inter-sentence silence is deliberately absent: gaps are AudioAssembler's job
        # now, so the engine must not add its own (issue #64).
        assert "--noise_scale" in cmd
        assert "--noise_w" in cmd
        assert "--sentence_silence" not in cmd

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_generate_with_command_line_handles_timeout(self, mock_run: Mock, custom_piper_config: PiperConfig) -> None:
        """Should handle command timeout gracefully."""
        # Mock successful piper check
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(custom_piper_config)

        # Mock timeout on actual generation
        from subprocess import TimeoutExpired

        mock_run.side_effect = TimeoutExpired("piper", 30)

        result = provider.synthesize("Test text")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
        assert "timed out after 30 seconds" in str(result.error.details)

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_generate_with_command_line_handles_command_failure(
        self, mock_run: Mock, custom_piper_config: PiperConfig
    ) -> None:
        """Should handle command execution failures."""
        # Mock successful piper check
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(custom_piper_config)

        # Mock command failure
        mock_run.return_value = Mock(returncode=1, stderr="Model not found", stdout="", args=["piper"])

        result = provider.synthesize("Test text")

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == ErrorCode.TTS_ENGINE_ERROR
        assert "Piper command failed with code 1" in str(result.error.details)


class TestPiperTTSProviderAsyncGeneration:
    """Test async TTS generation functionality."""

    @pytest.mark.asyncio
    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    async def test_synthesize_async_calls_sync_method(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should call sync method in thread pool."""
        # Mock successful piper availability
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(basic_piper_config)

        # Mock sync method to avoid actual TTS generation
        segment = SynthesizedSegment(
            text="Test text", pcm=b"\x00\x00" * 10, sample_rate=100, sample_width=2, channels=1
        )
        with patch.object(provider, "synthesize") as mock_sync:
            mock_sync.return_value = Result.success([segment])

            result = await provider.synthesize_async("Test text")

            assert result.is_success
            assert result.value == [segment]
            mock_sync.assert_called_once_with("Test text")


class TestPiperTTSProviderSecureDownload:
    """Test secure model downloading functionality."""

    def test_secure_download_validates_url_format(self, basic_piper_config: PiperConfig) -> None:
        """Should validate URL format before downloading."""
        provider = PiperTTSProvider(basic_piper_config)

        invalid_urls = [
            "not-a-url",
            "ftp://example.com/file.onnx",
            "http://insecure.com/file.onnx",  # HTTP not allowed
            "",
        ]

        for invalid_url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid URL format|Only HTTPS URLs are allowed"):
                provider._secure_download(invalid_url, "/tmp/test_file")

    @patch("urllib.request.urlopen")
    def test_secure_download_success(
        self, mock_urlopen: Mock, basic_piper_config: PiperConfig, temp_models_dir: str
    ) -> None:
        """Should successfully download files with SSL verification."""
        # Pre-create model files so the constructor's auto-download is skipped
        (Path(temp_models_dir) / "en_US-lessac-medium.onnx").touch()
        (Path(temp_models_dir) / "en_US-lessac-medium.onnx.json").touch()
        provider = PiperTTSProvider(basic_piper_config)

        # Mock successful download
        mock_response = Mock()
        mock_response.status = 200
        mock_response.read.side_effect = [b"file_content", b""]  # First call returns content, second is EOF
        mock_urlopen.return_value.__enter__.return_value = mock_response

        destination = str(Path(temp_models_dir) / "test_model.onnx")
        provider._secure_download("https://example.com/model.onnx", destination)

        # Verify file was written
        assert Path(destination).exists()
        assert Path(destination).read_bytes() == b"file_content"

        # Verify SSL context was used
        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        assert "context" in call_args.kwargs
        assert call_args.kwargs["timeout"] == 30

    @patch("urllib.request.urlopen")
    def test_secure_download_handles_http_errors(self, mock_urlopen: Mock, basic_piper_config: PiperConfig) -> None:
        """Should handle HTTP error responses."""
        provider = PiperTTSProvider(basic_piper_config)

        # Mock HTTP 404 error
        mock_response = Mock()
        mock_response.status = 404
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with pytest.raises(Exception, match="Download failed"):
            provider._secure_download("https://example.com/missing.onnx", "/tmp/test")


class TestPiperTTSProviderModelManagement:
    """Test model downloading and management."""

    @patch.object(PiperTTSProvider, "_secure_download")
    def test_ensure_model_downloads_when_missing(self, mock_download: Mock, temp_models_dir: str) -> None:
        """Should download model files when they don't exist."""
        config = PiperConfig(
            model_name="en_US-ryan-medium",
            download_dir=temp_models_dir,
        )

        provider = PiperTTSProvider(config)
        # The constructor also attempts an auto-download; only count the explicit call
        mock_download.reset_mock()
        model_path, config_path = provider._ensure_model()

        # Verify download was called for both files
        assert mock_download.call_count == 2

        # Verify paths are correct
        assert model_path.endswith("en_US-ryan-medium.onnx")
        assert config_path.endswith("en_US-ryan-medium.onnx.json")

    def test_ensure_model_returns_existing_files(self, temp_models_dir: str) -> None:
        """Should return existing model files without downloading."""
        config = PiperConfig(
            model_name="en_US-lessac-medium",
            download_dir=temp_models_dir,
        )

        # Create existing model files
        model_file = Path(temp_models_dir) / "en_US-lessac-medium.onnx"
        config_file = Path(temp_models_dir) / "en_US-lessac-medium.onnx.json"
        model_file.touch()
        config_file.touch()

        provider = PiperTTSProvider(config)

        with patch.object(provider, "_secure_download") as mock_download:
            model_path, config_path = provider._ensure_model()

            # Should not download when files exist
            mock_download.assert_not_called()

            assert model_path == str(model_file)
            assert config_path == str(config_file)

    @patch.object(PiperTTSProvider, "_secure_download")
    def test_ensure_model_handles_download_failure(self, mock_download: Mock, temp_models_dir: str) -> None:
        """Should handle model download failures gracefully."""
        config = PiperConfig(
            model_name="en_US-lessac-medium",
            download_dir=temp_models_dir,
        )

        mock_download.side_effect = Exception("Network error")

        provider = PiperTTSProvider(config)

        with pytest.raises(Exception, match="Model download failed"):
            provider._ensure_model()


class TestPiperTTSProviderRealWorldScenarios:
    """Test real-world usage scenarios and edge cases."""

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_handles_very_long_text(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should handle very long text input with appropriate timeout."""
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(basic_piper_config)

        # Create long text (5000 characters)
        long_text = "This is a very long text. " * 200

        with patch.object(provider, "_synthesize_with_command_line") as mock_generate:
            mock_generate.return_value = [
                SynthesizedSegment(text=long_text, pcm=b"\x00\x00" * 10, sample_rate=100, sample_width=2, channels=1)
            ]

            result = provider.synthesize(long_text)

            # Should process without timeout errors
            mock_generate.assert_called_once_with(long_text)

    @patch("infrastructure.tts.piper_tts_provider.PIPER_VOICE_AVAILABLE", False)
    @patch("subprocess.run")
    def test_handles_unicode_text(self, mock_run: Mock, basic_piper_config: PiperConfig) -> None:
        """Should handle Unicode characters in text."""
        mock_run.return_value = Mock(returncode=0)
        provider = PiperTTSProvider(basic_piper_config)

        unicode_text = "Héllo wörld! 你好世界 🌍"

        with patch.object(provider, "_synthesize_with_command_line") as mock_generate:
            mock_generate.return_value = [
                SynthesizedSegment(text=unicode_text, pcm=b"\x00\x00" * 10, sample_rate=100, sample_width=2, channels=1)
            ]

            result = provider.synthesize(unicode_text)

            # Should process Unicode text
            mock_generate.assert_called_once_with(unicode_text)

    def test_thread_safety_with_multiple_instances(self, temp_models_dir: str) -> None:
        """Should handle multiple provider instances safely."""
        configs = [
            PiperConfig(model_name="en_US-lessac-medium", download_dir=temp_models_dir),
            PiperConfig(model_name="en_US-ryan-medium", download_dir=temp_models_dir),
        ]

        # Should be able to create multiple instances without conflicts
        providers = [PiperTTSProvider(config) for config in configs]

        assert len(providers) == 2
        assert providers[0].config.model_name != providers[1].config.model_name
        assert all(p.models_dir == temp_models_dir for p in providers)


class TestPiperSynthesisConfig:
    """Test that PiperConfig prosody settings reach Piper's SynthesisConfig.

    These settings were parsed from config but never applied - see issue #59.
    """

    @staticmethod
    def _provider(config: PiperConfig) -> PiperTTSProvider:
        """Build a provider without touching model files or the network."""
        with patch.object(PiperTTSProvider, "_check_piper_availability"):
            provider = PiperTTSProvider.__new__(PiperTTSProvider)
            provider.config = config
            return provider

    def test_synthesis_config_carries_all_prosody_settings(self, temp_models_dir: str) -> None:
        """Every prosody knob should be forwarded, with noise_w renamed correctly."""
        config = PiperConfig(
            download_dir=temp_models_dir,
            length_scale=1.25,
            noise_scale=0.5,
            noise_w=0.9,
            speaker_id=3,
        )

        syn_config = self._provider(config)._build_synthesis_config()

        assert syn_config.length_scale == 1.25
        assert syn_config.noise_scale == 0.5
        # Our `noise_w` is Piper's `noise_w_scale` - an easy rename to get wrong
        assert syn_config.noise_w_scale == 0.9
        assert syn_config.speaker_id == 3

    def test_synthesis_config_disables_per_chunk_normalization(self, temp_models_dir: str) -> None:
        """Piper normalizes per chunk, and a chunk is one sentence.

        Left on, a one-word sentence is peaked to the same level as a paragraph and the
        output pumps at every sentence seam. Measured on en_US-lessac-medium: peaks are
        [1.0, 1.0] with it on versus [0.70, 0.52] with it off.
        """
        config = PiperConfig(download_dir=temp_models_dir)

        syn_config = self._provider(config)._build_synthesis_config()

        assert syn_config.normalize_audio is False

    def test_synthesis_config_uses_config_defaults(self, temp_models_dir: str) -> None:
        """Defaults should pass through rather than being dropped to Piper's own."""
        config = PiperConfig(download_dir=temp_models_dir)

        syn_config = self._provider(config)._build_synthesis_config()

        assert syn_config.length_scale == config.length_scale
        assert syn_config.noise_scale == config.noise_scale
        assert syn_config.noise_w_scale == config.noise_w


class TestPiperSegmentProduction:
    """Test that the Python-library path returns one segment per sentence.

    Gap policy and container writing live in AudioAssembler now (issue #64); what
    the provider owes its caller is segments whose text matches their audio.
    """

    @staticmethod
    def _provider(config: PiperConfig, chunks_per_call: int = 1, frames: int = 100) -> PiperTTSProvider:
        """Provider wired to a fake voice yielding `chunks_per_call` chunks per call."""
        with patch.object(PiperTTSProvider, "_check_piper_availability"):
            provider = PiperTTSProvider.__new__(PiperTTSProvider)
        provider.config = config
        provider.segmenter = TextSegmenter()

        def fake_synthesize(text: str, syn_config: object) -> object:
            return iter([_piper_chunk(frames) for _ in range(chunks_per_call)])

        voice = Mock()
        voice.synthesize.side_effect = fake_synthesize
        provider.voice_instance = voice  # type: ignore[assignment]
        return provider

    def test_one_segment_per_sentence_with_matching_text(self, temp_models_dir: str) -> None:
        """Each segment must carry the exact sentence that produced its audio."""
        provider = self._provider(PiperConfig(download_dir=temp_models_dir))

        segments = provider._synthesize_with_python_lib("First sentence. Second sentence. Third sentence.")

        assert [s.text for s in segments] == ["First sentence.", "Second sentence.", "Third sentence."]

    def test_segment_duration_is_measured_from_samples(self, temp_models_dir: str) -> None:
        """Duration comes from the sample count, not an estimate."""
        provider = self._provider(PiperConfig(download_dir=temp_models_dir), frames=50)

        segments = provider._synthesize_with_python_lib("One sentence.")

        # 50 frames at 100 Hz
        assert segments[0].duration == pytest.approx(0.5)

    def test_sub_chunks_are_joined_into_one_segment(self, temp_models_dir: str) -> None:
        """If Piper splits a sentence further, the pieces rejoin under the text we asked for."""
        provider = self._provider(PiperConfig(download_dir=temp_models_dir), chunks_per_call=3, frames=50)

        segments = provider._synthesize_with_python_lib("One sentence.")

        assert len(segments) == 1
        assert segments[0].text == "One sentence."
        assert segments[0].duration == pytest.approx(1.5)  # 3 x 50 frames at 100 Hz

    def test_provider_writes_no_silence_of_its_own(self, temp_models_dir: str) -> None:
        """Inter-sentence gaps are AudioAssembler's job; the engine adds none (issue #64)."""
        config = PiperConfig(download_dir=temp_models_dir)
        provider = self._provider(config, frames=100)

        segments = provider._synthesize_with_python_lib("One sentence.")

        # Exactly the speech frames, no trailing silence
        assert len(segments[0].pcm) == 100 * 2

    def test_zero_chunks_reports_the_real_error(self, temp_models_dir: str) -> None:
        """The failure must name the cause, not a wave-header side effect."""
        with patch.object(PiperTTSProvider, "_check_piper_availability"):
            provider = PiperTTSProvider.__new__(PiperTTSProvider)
        provider.config = PiperConfig(download_dir=temp_models_dir)
        provider.segmenter = TextSegmenter()
        voice = Mock()
        voice.synthesize.return_value = iter([])
        provider.voice_instance = voice  # type: ignore[assignment]

        with patch.object(provider, "_synthesize_with_command_line", side_effect=Exception("cli disabled")):
            try:
                provider._synthesize_with_python_lib("text")
            except Exception as exc:
                assert "channels not specified" not in str(exc.__context__)
                assert "no audio chunks" in str(exc.__context__)
