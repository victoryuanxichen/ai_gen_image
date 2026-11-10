"""Image decoding shared by training, evaluation and serving.

Every image enters the system through `open_image`, so the three stages cannot
drift apart in how they treat EXIF rotation, transparency or colour modes.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Union

from PIL import Image, ImageOps

# Refuse absurdly large images instead of letting them exhaust memory.
MAX_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ImageSource = Union[str, Path, bytes, bytearray]


class ImageError(ValueError):
    """The input could not be decoded as a usable image."""


def _to_rgb(im: Image.Image) -> Image.Image:
    """Convert any PIL mode to RGB, flattening transparency onto white."""
    if im.mode == "RGB":
        return im
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (
        im.mode == "P" and "transparency" in im.info
    )
    if has_alpha:
        rgba = im.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return im.convert("RGB")


def open_image(src: ImageSource) -> tuple[Image.Image, str]:
    """Decode `src` (a path or raw bytes) to an RGB image.

    Returns the image and its container format in upper case ("JPEG", "PNG", ...).
    Raises ImageError for anything that cannot be decoded.
    """
    try:
        with warnings.catch_warnings():
            # Treat PIL's decompression-bomb warning as a hard failure.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if isinstance(src, (bytes, bytearray)):
                im = Image.open(io.BytesIO(bytes(src)))
            else:
                im = Image.open(src)
            fmt = (im.format or "").upper()
            if fmt == "MPO":  # multi-picture JPEG written by many phones
                fmt = "JPEG"
            if getattr(im, "is_animated", False):
                im.seek(0)
            im = ImageOps.exif_transpose(im)
            im = _to_rgb(im)
            im.load()
    except ImageError:
        raise
    except Exception as exc:  # PIL raises many unrelated exception types
        raise ImageError(f"cannot decode image: {exc}") from exc
    if im.width < 8 or im.height < 8:
        raise ImageError(f"image too small: {im.width}x{im.height}")
    return im, fmt


def probe_image(path: Union[str, Path]) -> tuple[str, int, int]:
    """Read format and size from the file header without decoding pixels."""
    with Image.open(path) as im:
        fmt = (im.format or "").upper()
        if fmt == "MPO":
            fmt = "JPEG"
        return fmt, im.width, im.height


def encode_jpeg(im: Image.Image, quality: int) -> Image.Image:
    """Round-trip an RGB image through in-memory JPEG compression."""
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def harmonize(im: Image.Image, fmt: str, jpeg_quality: int | None) -> Image.Image:
    """Give lossless inputs the same compression history as JPEG inputs.

    Public benchmarks store real photos as JPEG and generated images as PNG. A
    detector trained on the raw files learns that shortcut instead of generation
    artifacts (Grommelt et al., 2024). Re-encoding every non-JPEG input at a high
    quality removes the cue. The same rule runs in training, evaluation and
    serving. Pass `jpeg_quality=None` to disable.
    """
    if jpeg_quality is None or fmt == "JPEG":
        return im
    return encode_jpeg(im, jpeg_quality)
