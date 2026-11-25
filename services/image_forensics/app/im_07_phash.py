from io import BytesIO
import time
from typing import Any
import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps


def compute_phash(
    file_bytes: bytes,
    hash_size: int = 8,
    work_size: int = 128,
) -> dict[str, Any]:
    """
    Compute a deterministic 64-bit perceptual hash (pHash) for an image.

    `hash_size=8` produces an 8x8 = 64-bit hash (16 hex characters).
    Downsamples to `work_size x work_size` before hashing so execution stays fast (< 50 ms).
    """
    if not file_bytes:
        raise ValueError("Cannot compute pHash from empty image bytes.")

    start_perf = time.perf_counter()

    with Image.open(BytesIO(file_bytes)) as img:
        oriented_img = ImageOps.exif_transpose(img)
        rgb_img = oriented_img.convert("RGB")

        np_rgb = np.asarray(rgb_img, dtype=np.uint8)
        gray_np = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)

        if work_size > 0 and max(gray_np.shape) > work_size:
            gray_np = cv2.resize(
                gray_np,
                (work_size, work_size),
                interpolation=cv2.INTER_AREA,
            )

        gray_pil = Image.fromarray(gray_np, mode="L")

    phash_obj = imagehash.phash(gray_pil, hash_size=hash_size)
    hex_digest = str(phash_obj).lower()
    int_value = int(hex_digest, 16)
    bit_string = format(int_value, f"0{hash_size * hash_size}b")

    elapsed_ms = round((time.perf_counter() - start_perf) * 1000, 3)

    return {
        "phash_hex": hex_digest,
        "phash_int": int_value,
        "phash_bits": bit_string,
        "bit_length": len(bit_string),
        "elapsed_ms": elapsed_ms,
        "within_50ms": elapsed_ms < 50.0,
    }
    
def phash_hamming_distance(hex_a: str, hex_b: str) -> int:
    """Return number of differing bits (0 to 64) between two 64-bit hex pHashes."""
    clean_a = hex_a.strip().lower()
    clean_b = hex_b.strip().lower()

    if len(clean_a) != len(clean_b):
        raise ValueError("Cannot compare pHashes of different lengths.")

    return (int(clean_a, 16) ^ int(clean_b, 16)).bit_count()