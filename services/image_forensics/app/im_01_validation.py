from io import BytesIO
from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS: dict[str, set[str]] = {
    "JPEG": {"image/jpeg", "image/jpg"},
    "PNG": {"image/png"},
    "WEBP": {"image/webp"},
}


class ImageValidationError(Exception):
    """Raised when an uploaded image fails MIME, magic-byte, or integrity checks."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _detect_magic_format(data: bytes) -> str | None:
    """Inspect leading binary bytes (magic numbers) to identify JPEG, PNG, or WebP."""
    if len(data) < 12:
        return None

    # JPEG files always start with 0xFF 0xD8 0xFF
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"

    # PNG files start with the exact 8-byte PNG signature
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    # WebP files are RIFF containers starting with 'RIFF' (bytes 0..3) and 'WEBP' (bytes 8..11)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "WEBP"

    return None


def validate_image_payload(file_bytes: bytes, content_type: str | None = None) -> dict:
    """
    Validate incoming image bytes and optional HTTP Content-Type header.

    Returns a small metadata dictionary if valid; raises ImageValidationError (HTTP 422)
    if empty, corrupted, disguised, or unsupported.
    """
    if not file_bytes:
        raise ImageValidationError("Uploaded file is empty.", status_code=422)

    detected_format = _detect_magic_format(file_bytes)
    if detected_format is None:
        raise ImageValidationError(
            "Invalid or unrecognized binary file signature. Expected JPEG, PNG, or WebP.",
            status_code=422,
        )

    normalized_mime = content_type.split(";")[0].strip().lower() if content_type else None
    allowed_mimes = SUPPORTED_FORMATS[detected_format]

    # If a Content-Type header was sent, make sure it matches the actual binary format
    if normalized_mime and normalized_mime not in allowed_mimes:
        raise ImageValidationError(
            f"Content-Type '{normalized_mime}' does not match detected binary format '{detected_format}'.",
            status_code=422,
        )

    # Verify the image isn't truncated or corrupted inside its data stream
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img.verify()

        # Re-open after verify() because verify() leaves the image object in an unusable state
        with Image.open(BytesIO(file_bytes)) as reopened:
            reopened.load()
            width, height = reopened.size
            pil_format = reopened.format
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(
            f"Corrupted or truncated image payload: {exc}",
            status_code=422,
        ) from exc

    return {
        "valid": True,
        "format": detected_format,
        "pil_format": pil_format,
        "mime_type": normalized_mime or next(iter(allowed_mimes)),
        "width": width,
        "height": height,
        "size_bytes": len(file_bytes),
    }