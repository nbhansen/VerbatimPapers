"""Error formatting utilities for user-facing error messages."""

import re


def clean_text_for_display(text: str) -> str:
    """Remove pause markers from text for display."""
    text = re.sub(r"\.{3,}", "", text)  # Remove ... sequences
    text = re.sub(r"\(\s*\)", "", text)  # Remove ( ) sequences
    text = re.sub(r"\s+", " ", text)  # Clean up multiple spaces

    return text.strip()


def get_processing_stage_error(
    stage: str,
    error: Exception,
    filename: str | None = None,
) -> str:
    """Get error message for specific processing stages with context."""
    file_info = f" for '{filename}'" if filename else ""

    stage_messages = {
        "text_extraction": f"Failed to extract text from PDF{file_info}",
        "text_processing": f"Failed to process and clean text{file_info}",
        "audio_generation": f"Failed to generate audio{file_info}",
        "file_combination": f"Failed to create final audio file{file_info}",
        "file_validation": f"File validation failed{file_info}",
        "configuration": f"Configuration error{file_info}",
    }

    base_message = stage_messages.get(stage, f"Processing failed at {stage}{file_info}")
    error_detail = str(error)

    # Add specific suggestions based on stage and error type
    if stage == "text_extraction" and ("password" in error_detail.lower() or "encrypted" in error_detail.lower()):
        return f"{base_message}: The PDF is password-protected. Please remove the password and try again."
    elif stage == "audio_generation" and "model" in error_detail.lower():
        return f"{base_message}: Voice model unavailable. Check your internet connection for automatic download."
    elif stage == "file_validation" and "size" in error_detail.lower():
        return f"{base_message}: File is too large. Try compressing or splitting the PDF."
    else:
        return f"{base_message}: {error_detail}"
