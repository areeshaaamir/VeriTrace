from io import BytesIO
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.im_01_validation import ImageValidationError, validate_image_payload
from app.im_02_normalization import normalize_for_inference
from app.im_03_exif import extract_exif_metadata
from app.im_07_phash import compute_phash, phash_hamming_distance


def _make_image_bytes(
    fmt: str = "PNG",
    mode: str = "RGB",
    size: tuple[int, int] = (320, 240),
    color=(120, 180, 240),
    draw_shape: bool = False,
) -> bytes:
    """Helper to generate valid image bytes in memory for tests."""
    if mode == "RGBA":
        img = Image.new("RGBA", size, (*color[:3], 128))
    elif mode == "LA":
        img = Image.new("LA", size, (150, 128))
    else:
        img = Image.new("RGB", size, color[:3])

    if draw_shape:
        drawer = ImageDraw.Draw(img)
        shape_fill = 20 if mode == "LA" else (20, 20, 20)
        drawer.rectangle(
            [size[0] // 4, size[1] // 4, (size[0] * 3) // 4, (size[1] * 3) // 4],
            fill=shape_fill,
        )

    buf = BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(buf, **save_kwargs)
    return buf.getvalue()


# --- IM-01: Validation & HTTP 422 ---


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("WEBP", "image/webp"),
    ],
)
def test_im01_valid_formats_pass(fmt: str, mime: str) -> None:
    raw = _make_image_bytes(fmt=fmt, mode="RGB", size=(160, 120))
    result = validate_image_payload(raw, content_type=mime)

    assert result["valid"] is True
    assert result["format"] == fmt
    assert result["width"] == 160
    assert result["height"] == 120


def test_im01_disguised_file_raises_422() -> None:
    fake_jpg = b"this is not a real jpeg even if saved as .jpg"
    with pytest.raises(ImageValidationError) as exc_info:
        validate_image_payload(fake_jpg, content_type="image/jpeg")

    assert exc_info.value.status_code == 422


def test_im01_mime_mismatch_raises_422() -> None:
    png_bytes = _make_image_bytes(fmt="PNG", mode="RGB")
    with pytest.raises(ImageValidationError) as exc_info:
        validate_image_payload(png_bytes, content_type="image/jpeg")

    assert exc_info.value.status_code == 422


def test_im01_truncated_jpeg_raises_422() -> None:
    valid_jpg = _make_image_bytes(fmt="JPEG", mode="RGB", size=(400, 400), draw_shape=True)
    truncated = valid_jpg[: len(valid_jpg) // 2]

    with pytest.raises(ImageValidationError) as exc_info:
        validate_image_payload(truncated, content_type="image/jpeg")

    assert exc_info.value.status_code == 422


# --- IM-02: Normalization & Original Preservation ---


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA"])
def test_im02_normalizes_to_rgb_and_preserves_original(mode: str) -> None:
    original = _make_image_bytes(fmt="PNG", mode=mode, size=(500, 300), draw_shape=True)
    out = normalize_for_inference(original, target_size=(224, 224))

    assert out["original_bytes"] == original
    assert out["mode"] == "RGB"
    assert out["normalized_size"] == (224, 224)


# --- IM-03: EXIF Extraction ---


def test_im03_extracts_exif_when_present() -> None:
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    exif = img.getexif()
    # 271 = Make, 272 = Model
    exif[271] = "TestCam"
    exif[272] = "Model X1"

    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    report = extract_exif_metadata(buf.getvalue())
    assert report["has_exif"] is True
    assert report["camera"]["make"] == "TestCam"
    assert report["camera"]["model"] == "Model X1"


def test_im03_handles_missing_exif_cleanly() -> None:
    png_bytes = _make_image_bytes(fmt="PNG", mode="RGB")
    report = extract_exif_metadata(png_bytes)

    assert report["has_exif"] is False
    assert report["camera"]["make"] is None
    assert report["camera"]["model"] is None
    assert report["capture_settings"]["focal_length_mm"] is None


# --- IM-07: Deterministic 64-Bit pHash (< 50 ms) ---


def test_im07_deterministic_64bit_phash_under_50ms() -> None:
    img_bytes = _make_image_bytes(fmt="PNG", mode="RGB", size=(512, 512), draw_shape=True)
    
    _ = compute_phash(img_bytes)

    first = compute_phash(img_bytes)
    second = compute_phash(img_bytes)

    assert first["bit_length"] == 64
    assert len(first["phash_hex"]) == 16
    assert first["phash_hex"] == second["phash_hex"]
    assert first["within_50ms"] is True
    assert phash_hamming_distance(first["phash_hex"], second["phash_hex"]) == 0