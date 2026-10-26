# utils.py - Pure utility functions with no domain dependencies
from collections.abc import Mapping

from domain.models import PageRange

# Flask's ImmutableMultiDict is a Mapping, as is the plain dict the background job gets
FormData = Mapping[str, str]


def allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed file extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}


def _parse_page_number(raw: str, label: str) -> int:
    """Parse a 1-based page number from form input."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{label} must be a whole number, got '{raw}'") from None
    if value < 1:
        raise ValueError(f"{label} must be 1 or greater, got {value}")
    return value


def parse_page_range_from_form(form: FormData) -> PageRange:
    """Parse page range from Flask form data.

    Raises:
        ValueError: if a page field is not a positive integer, or the start
            page comes after the end page.
    """
    use_page_range = form.get("use_page_range") == "on"

    if not use_page_range:
        return PageRange()

    start_page: int | None = None
    end_page: int | None = None

    start_page_str = str(form.get("start_page", "")).strip()
    end_page_str = str(form.get("end_page", "")).strip()

    if start_page_str:
        start_page = _parse_page_number(start_page_str, "start page")

    if end_page_str:
        end_page = _parse_page_number(end_page_str, "end page")

    if start_page is not None and end_page is not None and start_page > end_page:
        raise ValueError(f"start page ({start_page}) cannot be after end page ({end_page})")

    return PageRange(start_page=start_page, end_page=end_page)


def parse_plain_english_from_form(form: FormData) -> bool:
    """Parse plain English conversion setting from Flask form data."""
    return bool(form.get("enable_plain_english") == "on")
