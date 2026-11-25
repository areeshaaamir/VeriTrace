from io import BytesIO
from typing import Any
from PIL import ExifTags, Image


# Standard EXIF tag IDs for the fields we want
EXIF_TAG_IDS: dict[str, int] = {
    "make": ExifTags.Base.Make,
    "model": ExifTags.Base.Model,
    "focal_length": ExifTags.Base.FocalLength,
    "exposure_time": ExifTags.Base.ExposureTime,
    "f_number": ExifTags.Base.FNumber,
    "iso": ExifTags.Base.ISOSpeedRatings,
}


def _to_json_safe(value: Any) -> Any:
    """
    Convert Pillow EXIF values (including IFDRational or bytes) into clean,
    serializable Python types without crashing.
    """
    if value is None:
        return None

    # Pillow's IFDRational has numerator and denominator attributes
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        if value.denominator == 0:
            return None
        return round(float(value.numerator) / float(value.denominator), 4)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00").strip()

    if isinstance(value, str):
        cleaned = value.strip("\x00").strip()
        return cleaned if cleaned else None

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        # Sometimes ISO or similar tags come back as a 1-element tuple/list
        if len(value) == 1:
            return _to_json_safe(value[0])
        return [_to_json_safe(item) for item in value]

    return value


def _format_exposure_time(raw_exposure: Any) -> dict[str, Any]:
    """Return both a human-readable fraction/string and numeric seconds when available."""
    if raw_exposure is None:
        return {"shutter_speed": None, "seconds": None}

    if hasattr(raw_exposure, "numerator") and hasattr(raw_exposure, "denominator"):
        num = int(raw_exposure.numerator)
        den = int(raw_exposure.denominator)
        if den == 0:
            return {"shutter_speed": None, "seconds": None}
        seconds = round(num / den, 6)
        label = f"{num}/{den}s" if num == 1 and den > 1 else f"{seconds}s"
        return {"shutter_speed": label, "seconds": seconds}

    numeric = _to_json_safe(raw_exposure)
    if isinstance(numeric, (int, float)):
        return {"shutter_speed": f"{numeric}s", "seconds": numeric}

    return {"shutter_speed": str(numeric), "seconds": None}


def extract_exif_metadata(file_bytes: bytes) -> dict[str, Any]:
    """
    Extract camera make, model, focal length, and exposure settings from EXIF headers
    when present. Returns structured fields with None values when EXIF is absent.
    """
    empty_report = {
        "has_exif": False,
        "camera": {
            "make": None,
            "model": None,
        },
        "capture_settings": {
            "focal_length_mm": None,
            "exposure": {
                "shutter_speed": None,
                "seconds": None,
            },
            "aperture_f_number": None,
            "iso": None,
        },
    }

    if not file_bytes:
        return empty_report

    try:
        with Image.open(BytesIO(file_bytes)) as img:
            exif = img.getexif()
            if not exif:
                return empty_report

            # EXIF IFD (Image File Directory) holds detailed shooting tags like focal length & exposure
            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif) if hasattr(exif, "get_ifd") else {}

            def get_tag(field_key: str) -> Any:
                tag_id = EXIF_TAG_IDS[field_key]
                if tag_id in exif:
                    return exif.get(tag_id)
                if exif_ifd and tag_id in exif_ifd:
                    return exif_ifd.get(tag_id)
                return None

            make = _to_json_safe(get_tag("make"))
            model = _to_json_safe(get_tag("model"))
            focal_length = _to_json_safe(get_tag("focal_length"))
            exposure_info = _format_exposure_time(get_tag("exposure_time"))
            f_number = _to_json_safe(get_tag("f_number"))
            iso = _to_json_safe(get_tag("iso"))

            has_any_data = any(
                val is not None
                for val in (
                    make,
                    model,
                    focal_length,
                    exposure_info["seconds"],
                    f_number,
                    iso,
                )
            )

            return {
                "has_exif": has_any_data,
                "camera": {
                    "make": make,
                    "model": model,
                },
                "capture_settings": {
                    "focal_length_mm": focal_length,
                    "exposure": exposure_info,
                    "aperture_f_number": f_number,
                    "iso": iso,
                },
            }
    except Exception:
        # Gracefully fall back if metadata is malformed or unreadable
        return empty_report