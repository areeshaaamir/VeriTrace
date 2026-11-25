from io import BytesIO
from PIL import Image, ImageOps


DEFAULT_INFERENCE_SIZE: tuple[int, int] = (224, 224)


def _flatten_alpha_to_rgb(img: Image.Image, background_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Safely convert an image with an alpha/transparency channel (such as RGBA or LA)
    into a 3-channel RGB image without color distortion.
    """
    # Ensure transparency info is represented as an explicit alpha channel
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*background_color, 255))
        composited = Image.alpha_composite(background, rgba)
        return composited.convert("RGB")

    return img.convert("RGB")


def normalize_for_inference(
    file_bytes: bytes,
    target_size: tuple[int, int] = DEFAULT_INFERENCE_SIZE,
) -> dict:
    """
    Keep original bytes untouched for Error Level Analysis (ELA), while producing
    a clean 3-channel RGB image resized to `target_size` for neural inference (e.g., EfficientNet).
    """
    # 1. Preserve the exact original bytes untouched for ELA
    original_bytes = file_bytes

    # 2. Load into a fresh Pillow image without mutating `original_bytes`
    with Image.open(BytesIO(original_bytes)) as raw_img:
        # Apply EXIF orientation if present so portrait/rotated images aren't sideways
        oriented_img = ImageOps.exif_transpose(raw_img)
        original_size = oriented_img.size

        # 3. Handle alpha channels cleanly before converting to 3-channel RGB
        rgb_img = _flatten_alpha_to_rgb(oriented_img)

        # 4. Resize to consistent tensor dimensions (e.g., 224x224 for EfficientNet)
        resized_rgb_img = rgb_img.resize(target_size, Image.Resampling.BICUBIC)

        # Detach from the file buffer so the image remains usable after the `with` block closes
        normalized_image = resized_rgb_img.copy()

    return {
        "original_bytes": original_bytes,
        "original_size": original_size,
        "normalized_image": normalized_image,
        "normalized_size": normalized_image.size,
        "mode": normalized_image.mode,  # Always "RGB" (3 channels)
    }