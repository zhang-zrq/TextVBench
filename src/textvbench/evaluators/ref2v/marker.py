"""Marker and reference-label image rendering."""

from __future__ import annotations

import base64
import colorsys
import io
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .core import Candidate


COLORS = (
    (230, 72, 72), (54, 122, 240), (33, 166, 102), (229, 145, 32),
    (157, 78, 221), (0, 151, 167), (214, 55, 152), (104, 126, 44),
)


def _color(index: int) -> tuple[int, int, int]:
    if index < len(COLORS):
        return COLORS[index]
    red, green, blue = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.82, 0.92)
    return round(red * 255), round(green * 255), round(blue * 255)


def _font(image: Image.Image, font_path: str | Path | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    if not font_path:
        return None
    try:
        return ImageFont.truetype(str(font_path), max(12, image.width // 90))
    except OSError:
        return None


def as_image(value: Image.Image | str | Path | object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB").copy()
    if hasattr(value, "shape"):
        return Image.fromarray(value[:, :, ::-1]).convert("RGB")
    return Image.open(value).convert("RGB")


def resize_longest_edge(image: object, max_edge: int = 256) -> Image.Image:
    """Resize an image proportionally without upscaling it."""
    if max_edge < 1:
        raise ValueError("max_edge must be positive")
    result = as_image(image)
    longest = max(result.size)
    if longest <= max_edge:
        return result
    scale = max_edge / longest
    size = (max(1, round(result.width * scale)), max(1, round(result.height * scale)))
    return result.resize(size, Image.Resampling.LANCZOS)


def render_marker(frame: object, candidates: Iterable[Candidate], font_path: str | Path | None = None) -> Image.Image:
    image = as_image(frame)
    draw = ImageDraw.Draw(image)
    font = _font(image, font_path)
    for index, candidate in enumerate(candidates):
        if not candidate.bbox:
            continue
        color = _color(index)
        points = [(int(round(x)), int(round(y))) for x, y in candidate.bbox]
        draw.line(points + [points[0]], fill=color, width=max(2, image.width // 500), joint="curve")
        left = min(x for x, _ in points)
        top = min(y for _, y in points)
        bounds = draw.textbbox((0, 0), candidate.id, font=font)
        width = bounds[2] - bounds[0] + 8
        height = bounds[3] - bounds[1] + 5
        x = max(0, min(left, image.width - width))
        y = top - height if top >= height else min(top, image.height - height)
        draw.rectangle((x, y, x + width, y + height), fill="white")
        draw.text((x + 4, y + 2), candidate.id, fill=color, font=font)
    return image


def render_reference_label(image: object, ref_id: str, font_path: str | Path | None = None) -> Image.Image:
    """Add a ref label in a top border band without covering source pixels."""
    result = as_image(image)
    font = _font(result, font_path)
    probe = ImageDraw.Draw(result)
    bounds = probe.textbbox((0, 0), ref_id, font=font)
    band_height = max(30, bounds[3] - bounds[1] + 14)
    source_height = result.height
    labeled = Image.new("RGB", (result.width, source_height + band_height), "white")
    labeled.paste(result, (0, band_height))
    draw = ImageDraw.Draw(labeled)
    width = min(labeled.width, bounds[2] - bounds[0] + 20)
    draw.rectangle((0, 0, width, max(0, band_height - 1)), fill="white")
    draw.text((10, 7), ref_id, fill=(20, 20, 20), font=font)
    return labeled


def image_data_url(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=image_format, quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"
