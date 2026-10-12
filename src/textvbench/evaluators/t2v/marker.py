"""Marker-image rendering for VLM candidate selection."""

from __future__ import annotations

import io
import colorsys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .core import Candidate


COLORS = (
    (230, 72, 72), (54, 122, 240), (33, 166, 102), (229, 145, 32),
    (157, 78, 221), (0, 151, 167), (214, 55, 152), (104, 126, 44),
)


def _color(index: int) -> tuple[int, int, int]:
    """Generate stable distinct hues when the candidate cap exceeds the palette."""
    if index < len(COLORS):
        return COLORS[index]
    red, green, blue = colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.82, 0.92)
    return round(red * 255), round(green * 255), round(blue * 255)


def render_marker(frame, candidates: Iterable[Candidate], font_path: str | Path | None = None) -> Image.Image:
    """Draw candidate polygons and IDs while preserving the complete frame."""
    if hasattr(frame, "shape"):
        # Avoid a hard dependency on cv2 in this module; OpenCV frames are BGR.
        image = Image.fromarray(frame[:, :, ::-1])
    elif isinstance(frame, Image.Image):
        image = frame.copy()
    else:
        image = Image.open(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(str(font_path), max(12, image.width // 90))
        except OSError:
            font = None
    for index, candidate in enumerate(candidates):
        if not candidate.bbox:
            continue
        color = _color(index)
        points = [(int(round(x)), int(round(y))) for x, y in candidate.bbox]
        draw.line(points + [points[0]], fill=color, width=max(2, image.width // 500), joint="curve")
        left = min(x for x, _ in points)
        top = min(y for _, y in points)
        label = candidate.id
        bounds = draw.textbbox((0, 0), label, font=font)
        width, height = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 5
        x = max(0, min(left, image.width - width))
        y = top - height if top >= height else min(top, image.height - height)
        draw.rectangle((x, y, x + width, y + height), fill="white")
        draw.text((x + 4, y + 2), label, fill=color, font=font)
    return image


def marker_data_url(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=image_format, quality=92)
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
