# infrastructure/audio/ffmpeg_audio_encoder.py - ffmpeg adapter for IAudioEncoder
"""MP3 encoding backed by the ffmpeg CLI.

Implements the domain's IAudioEncoder interface so the domain layer never
spawns processes. All errors are returned as Result[T].
"""

import logging
import subprocess  # nosec B404

from domain.errors import Result, audio_generation_error
from domain.interfaces import IAudioEncoder

logger = logging.getLogger(__name__)


class FfmpegAudioEncoder(IAudioEncoder):
    """Encodes WAV to MP3 by piping it through ffmpeg."""

    def __init__(self, bitrate: str = "192k", sample_rate: int = 44100, timeout_seconds: float = 300):
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.timeout_seconds = timeout_seconds

    def encode_mp3(self, wav: bytes, dest_path: str) -> Result[str]:
        """Encode WAV bytes to dest_path; the WAV goes in on stdin, so no temp file is written."""
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite if exists
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            self.bitrate,
            "-ar",
            str(self.sample_rate),
            dest_path,
        ]
        try:
            result = subprocess.run(cmd, input=wav, capture_output=True, timeout=self.timeout_seconds)  # nosec B603
        except FileNotFoundError:
            return Result.failure(audio_generation_error("ffmpeg not found; install it to produce MP3 output"))
        except subprocess.TimeoutExpired:
            return Result.failure(audio_generation_error(f"ffmpeg timed out after {self.timeout_seconds}s"))

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            logger.error("ffmpeg failed (exit %d): %s", result.returncode, stderr)
            return Result.failure(audio_generation_error(f"ffmpeg conversion failed: {stderr[-500:]}"))

        return Result.success(dest_path)
