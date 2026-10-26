"""FfmpegAudioEncoder: WAV bytes in, MP3 file out, failures as Result."""

import io
from pathlib import Path
import shutil
from unittest.mock import patch
import wave

import pytest

from infrastructure.audio.ffmpeg_audio_encoder import FfmpegAudioEncoder


def _wav(seconds: float = 0.5, sample_rate: int = 22050) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(int(seconds * sample_rate) * 2))
    return buffer.getvalue()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encodes_wav_to_mp3(tmp_path: Path) -> None:
    """The MP3 is written where asked, and nothing else is left behind."""
    dest = tmp_path / "out.mp3"

    result = FfmpegAudioEncoder().encode_mp3(_wav(), str(dest))

    assert result.is_success and result.value == str(dest)
    assert dest.read_bytes()[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    assert list(tmp_path.iterdir()) == [dest]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_invalid_input_is_a_failure(tmp_path: Path) -> None:
    """Garbage in is reported with ffmpeg's error, not raised."""
    result = FfmpegAudioEncoder().encode_mp3(b"not audio", str(tmp_path / "out.mp3"))

    assert result.is_failure and result.error is not None
    assert "ffmpeg conversion failed" in (result.error.details or "")


def test_missing_ffmpeg_is_a_failure(tmp_path: Path) -> None:
    """No ffmpeg on PATH is reported with an install hint."""
    with patch("infrastructure.audio.ffmpeg_audio_encoder.subprocess.run", side_effect=FileNotFoundError):
        result = FfmpegAudioEncoder().encode_mp3(_wav(), str(tmp_path / "out.mp3"))

    assert result.is_failure and result.error is not None
    assert "ffmpeg not found" in (result.error.details or "")
