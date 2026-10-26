"""Manual check of the real Gemini API, which CI can't reach without a key.

Usage: uv run python scripts/gemini_smoke.py [path/to/config.yaml]

Uses secrets.google_ai_api_key from the config. Runs one short LLM cleaning call
and a two-chunk Gemini TTS conversion through the app's own wiring, then checks
the MP3's real length against the read-along timeline. Costs a few API calls.
Output files are deleted afterwards.
"""

import dataclasses
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.composition import Services
from application.config.system_config import SystemConfig
from domain.config.tts_config import TTSEngine


def main() -> int:
    """Run the checks and return a process exit code."""
    config = SystemConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    if config.gemini is None or not config.gemini.api_key:
        print("FAIL: no secrets.google_ai_api_key in the config")
        return 1
    # One chunk per sentence below, so the run makes two TTS requests and has a seam between them
    services = Services(
        dataclasses.replace(
            config,
            tts=dataclasses.replace(config.tts, engine=TTSEngine.GEMINI),
            text_processing=dataclasses.replace(config.text_processing, audio_target_chunk_size=60),
        )
    )
    ok = True

    llm = services.llm_provider
    assert llm is not None
    cleaned = llm.generate_content("Reply with exactly: The quick brown fox.")
    print(f"LLM cleaning ({config.llm.model_name}):", "OK" if cleaned.is_success else f"FAIL {cleaned.error}")
    ok &= cleaned.is_success

    ready = services.tts_engine.check_ready()
    print(f"TTS ready ({config.gemini.model_name}):", "OK" if ready.is_success else f"FAIL {ready.error}")
    ok &= ready.is_success

    chunks = ["This is the first part of a short test.", "And this is the second part, read after a pause."]
    result = services.audio_engine.generate(chunks, "gemini_smoke")
    if result.is_failure or result.value is None or result.value.timing_data is None:
        print("FAIL TTS conversion:", result.error)
        return 1

    mp3 = Path(services.file_manager.get_output_dir()) / "gemini_smoke.mp3"
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(probe.stdout.strip())
        timeline = result.value.timing_data.total_duration
        segments = len(result.value.timing_data.text_segments)
        drift_ok = abs(duration - timeline) < 0.15
        print(
            f"TTS conversion: {segments} segments, MP3 {duration:.2f}s, timeline {timeline:.2f}s,",
            "OK" if drift_ok else "FAIL (drift)",
        )
        ok &= drift_ok and segments == 2
    finally:
        mp3.unlink(missing_ok=True)

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
