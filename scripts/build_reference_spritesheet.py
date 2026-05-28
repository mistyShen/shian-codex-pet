#!/usr/bin/env python3
"""Build a first-pass Codex pet spritesheet from the Shian reference board.

This is a deterministic package builder for the user-provided reference art.
It extracts the chibi character from the reference crops, creates restrained
motion frames, composes the 8x9 Codex pet atlas, and writes package files.
"""

from __future__ import annotations

import json
import math
import shutil
import colorsys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "references"
RUN_DIR = ROOT / "runs" / "shian-helper-reference-build"
FINAL_DIR = RUN_DIR / "final"
QA_DIR = RUN_DIR / "qa"
FRAMES_DIR = RUN_DIR / "frames"
PACKAGE_DIR = ROOT / "package"
CODEX_PACKAGE_DIR = Path.home() / ".codex" / "pets" / "shian-helper"
# Wide enough to include both twin-tail tips, but stops above the pose label.
# Non-character panel marks are filtered out by crop_subject().
FRONT_POSE_BOX = (348, 98, 578, 420)

CELL_W = 192
CELL_H = 208
COLUMNS = 8
PREVIEW_GIF_FRAME_MS = 135
ROWS = [
    ("idle", 8),
    ("running-right", 8),
    ("running-left", 8),
    ("waving", 4),
    ("jumping", 5),
    ("failed", 8),
    ("waiting", 6),
    ("running", 8),
    ("review", 6),
]

POSES_DIR = RUN_DIR / "poses"
POSE_SOURCE_FILES = {
    "idle": "user_states/idle.png",
    "running-right": "user_states/running-right.png",
    "running-left": "user_states/running-left.png",
    "waving": "user_states/waving.png",
    "jumping": "user_states/jumping.png",
    "failed": "user_states/failed.png",
    "waiting": "user_states/waiting.png",
    "running": "user_states/running.png",
    "review": "user_states/review.png",
}
POSE_SOURCE_BOXES = {}
POSE_SUBJECT_PAD = {
    "idle": 14,
    "running-right": 16,
    "running-left": 16,
    "waving": 12,
    "jumping": 18,
    "failed": 12,
    "waiting": 12,
    "running": 14,
    "review": 12,
}
POSE_TARGET_H = {
    "idle": 184,
    "running-right": 184,
    "running-left": 184,
    "waving": 184,
    "jumping": 184,
    "failed": 184,
    "waiting": 184,
    "running": 184,
    "review": 184,
}
POSE_TARGET_W = {
    "idle": 176,
    "running-right": 176,
    "running-left": 176,
    "waving": 176,
    "jumping": 176,
    "failed": 176,
    "waiting": 176,
    "running": 176,
    "review": 176,
}
POSE_MASK_GROW = {
    "idle": 5,
    "running-right": 3,
    "running-left": 3,
    "waving": 5,
    "jumping": 3,
    "failed": 3,
    "waiting": 1,
    "running": 3,
    "review": 1,
}
POSE_MASK_BLUR = {
    "idle": 0.30,
    "running-right": 0.18,
    "running-left": 0.18,
    "waving": 0.25,
    "jumping": 0.18,
    "failed": 0.18,
    "waiting": 0.08,
    "running": 0.18,
    "review": 0.08,
}
POSE_ALPHA_THRESHOLD = {
    "idle": 4,
    "running-right": 4,
    "running-left": 4,
    "waving": 4,
    "jumping": 4,
    "failed": 4,
    "waiting": 4,
    "running": 4,
    "review": 4,
}
POSE_TRIM_PAD = {
    "idle": 5,
    "running-right": 7,
    "running-left": 7,
    "waving": 5,
    "jumping": 8,
    "failed": 5,
    "waiting": 5,
    "running": 6,
    "review": 5,
}
POSE_CANVAS_PAD = {
    "idle": (8, 4),
    "running-right": (8, 4),
    "running-left": (8, 4),
    "waving": (8, 4),
    "jumping": (10, 4),
    "failed": (8, 4),
    "waiting": (10, 4),
    "running": (8, 4),
    "review": (10, 4),
}


@dataclass
class Motion:
    x: int = 0
    y: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotate: float = 0.0
    glow: float = 0.0
    opacity: float = 1.0
    hair_sway: float = 0.0
    skirt_sway: float = 0.0
    skirt_lift: float = 0.0
    cube_dx: float = 0.0
    cube_dy: float = 0.0
    cube_spin: float = 0.0
    blink: float = 0.0
    focus: float = 0.0
    eye_shift: float = 0.0
    brow: float = 0.0
    blush: float = 0.0
    sparkle: float = 0.0
    arm_raise: float = 0.0
    hug: float = 0.0
    run_stride: float = 0.0
    run_direction: int = 1
    crouch: float = 0.0
    sit: float = 0.0
    head_turn: float = 0.0
    head_nod: float = 0.0
    torso_breathe: float = 0.0
    torso_lean: float = 0.0
    left_hair_sway: float = 0.0
    right_hair_sway: float = 0.0
    skirt_fan: float = 0.0


def ensure_dirs() -> None:
    for path in (FINAL_DIR, QA_DIR, FRAMES_DIR, POSES_DIR, PACKAGE_DIR, CODEX_PACKAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_rgb(name: str) -> Image.Image:
    return Image.open(REF_DIR / name).convert("RGB")


def background_mask(image: Image.Image) -> Image.Image:
    """Return a foreground mask without deleting enclosed pale skin areas.

    A simple per-pixel chroma key is too aggressive for Shian because the skin
    tones are close to the off-white brief-board background. Instead, flood-fill
    only the light background pixels connected to the crop border. Pale pixels
    enclosed by hair, outlines, clothing, or props remain foreground.
    """

    rgb = image.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    samples = []
    for x in range(w):
        samples.append(px[x, 0])
        samples.append(px[x, h - 1])
    for y in range(h):
        samples.append(px[0, y])
        samples.append(px[w - 1, y])
    bg = tuple(sorted(c[i] for c in samples)[len(samples) // 2] for i in range(3))

    def is_background_like(x: int, y: int) -> bool:
        r, g, b = px[x, y]
        dist = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
        saturation = max(r, g, b) - min(r, g, b)
        bright = min(r, g, b) >= 226
        return bright and saturation <= 24 and dist <= 58

    background = Image.new("1", (w, h), 0)
    bp = background.load()
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if bp[x, y] == 0 and is_background_like(x, y):
                bp[x, y] = 1
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if bp[x, y] == 0 and is_background_like(x, y):
                bp[x, y] = 1
                q.append((x, y))

    while q:
        x, y = q.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if bp[nx, ny] == 0 and is_background_like(nx, ny):
                bp[nx, ny] = 1
                q.append((nx, ny))

    mask = Image.new("L", (w, h), 255)
    mp = mask.load()
    for y in range(h):
        for x in range(w):
            if bp[x, y]:
                mp[x, y] = 0

    # Smooth the silhouette edge without punching holes in enclosed light areas.
    mask = mask.filter(ImageFilter.MedianFilter(3))
    return mask


def connected_components(mask: Image.Image) -> list[tuple[int, tuple[int, int, int, int], Image.Image]]:
    """Find foreground components in a small binary mask."""

    binary = mask.point(lambda p: 255 if p > 0 else 0, "1")
    w, h = binary.size
    bp = binary.load()
    seen = set()
    out = []

    for sy in range(h):
        for sx in range(w):
            if (sx, sy) in seen or bp[sx, sy] == 0:
                continue
            q: deque[tuple[int, int]] = deque([(sx, sy)])
            seen.add((sx, sy))
            points = []
            min_x = max_x = sx
            min_y = max_y = sy
            while q:
                x, y = q.popleft()
                points.append((x, y))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if (nx, ny) in seen or bp[nx, ny] == 0:
                            continue
                        seen.add((nx, ny))
                        q.append((nx, ny))
            comp_mask = Image.new("L", (w, h), 0)
            cp = comp_mask.load()
            for x, y in points:
                cp[x, y] = 255
            out.append((len(points), (min_x, min_y, max_x + 1, max_y + 1), comp_mask))
    return sorted(out, reverse=True, key=lambda item: item[0])


def crop_subject(
    image: Image.Image,
    pad: int = 10,
    *,
    grow: int = 5,
    blur: float = 0.35,
    alpha_threshold: int = 8,
    merge_all_components: bool = False,
    min_component_area: int = 8,
) -> Image.Image:
    mask = background_mask(image)
    comps = connected_components(mask)
    if not comps:
        raise RuntimeError("no foreground components found")

    w, h = image.size

    if merge_all_components:
        merged = Image.new("L", image.size, 0)
        for area, _box, comp_mask in comps:
            if area < min_component_area:
                continue
            merged = ImageChops.lighter(merged, comp_mask)
    else:
        # The reference board crop includes panel rules and sometimes neighboring
        # pose slivers. The character is the only large connected foreground mass,
        # so keep that component alone instead of merging nearby board artifacts.
        _largest_area, _largest_box, merged = comps[0]
    if grow > 1:
        merged = merged.filter(ImageFilter.MaxFilter(grow))
    if blur > 0:
        merged = merged.filter(ImageFilter.GaussianBlur(blur))
    bbox = merged.point(lambda p: 255 if p > alpha_threshold else 0).getbbox()
    if not bbox:
        raise RuntimeError("foreground bbox missing")
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    rgba = image.convert("RGBA")
    rgba.putalpha(merged)
    return rgba.crop((x0, y0, x1, y1))


def apply_background_alpha(image: Image.Image, *, grow: int = 1, blur: float = 0.0) -> Image.Image:
    """Make the action template background transparent without cropping its canvas."""

    mask = background_mask(image)
    if grow > 1:
        mask = mask.filter(ImageFilter.MaxFilter(grow))
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def pose_color_stats(image: Image.Image) -> tuple[float, float, float]:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size
    luma_sum = 0.0
    luma_sq_sum = 0.0
    sat_sum = 0.0
    weight_sum = 0.0

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if a < 16:
                continue
            weight = a / 255.0
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sat = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            luma_sum += luma * weight
            luma_sq_sum += (luma * luma) * weight
            sat_sum += sat * weight
            weight_sum += weight

    if weight_sum == 0:
        return (0.0, 0.0, 0.0)

    mean_luma = luma_sum / weight_sum
    mean_sat = sat_sum / weight_sum
    variance = max(0.0, (luma_sq_sum / weight_sum) - (mean_luma * mean_luma))
    contrast = math.sqrt(variance)
    return (mean_luma, mean_sat, contrast)


def clampf(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_pose_colors(image: Image.Image, target_stats: tuple[float, float, float]) -> Image.Image:
    target_luma, target_sat, target_contrast = target_stats
    if pose_color_stats(image)[0] <= 0:
        return image

    def with_alpha(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
        out = rgb.convert("RGBA")
        out.putalpha(alpha)
        return out

    alpha = image.getchannel("A")
    rgb = image.convert("RGB")

    current_luma, _, _ = pose_color_stats(with_alpha(rgb, alpha))
    brightness = clampf(target_luma / max(current_luma, 1e-6), 0.72, 1.45)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)

    _, current_sat, _ = pose_color_stats(with_alpha(rgb, alpha))
    saturation = clampf(target_sat / max(current_sat, 1e-6), 0.72, 1.45) if current_sat > 0.001 else 1.0
    rgb = ImageEnhance.Color(rgb).enhance(saturation)

    current_luma, _, current_contrast = pose_color_stats(with_alpha(rgb, alpha))
    contrast = clampf(target_contrast / max(current_contrast, 1e-6), 0.78, 1.35) if current_contrast > 0.001 else 1.0
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)

    current_luma, _, _ = pose_color_stats(with_alpha(rgb, alpha))
    final_brightness = clampf(target_luma / max(current_luma, 1e-6), 0.88, 1.12)
    rgb = ImageEnhance.Brightness(rgb).enhance(final_brightness)

    return with_alpha(rgb, alpha)


def tune_pose_palette(image: Image.Image, state: str) -> Image.Image:
    if state != "running":
        return image

    alpha = image.getchannel("A")
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(0.98)

    px = rgb.load()
    w, h = rgb.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            px[x, y] = (
                max(0, min(255, round(r * 1.045))),
                max(0, min(255, round(g * 1.003))),
                max(0, min(255, round(b * 0.952))),
            )

    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def trim_alpha(image: Image.Image, pad: int = 4, *, alpha_threshold: int = 8) -> Image.Image:
    bbox = image.getchannel("A").point(lambda p: 255 if p > alpha_threshold else 0).getbbox()
    if not bbox:
        return image
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.width, x1 + pad)
    y1 = min(image.height, y1 + pad)
    return image.crop((x0, y0, x1, y1))


def fit_sprite(
    sprite: Image.Image,
    target_h: int = 162,
    target_w: int = 126,
    *,
    trim_pad: int = 4,
    trim_threshold: int = 8,
) -> Image.Image:
    sprite = trim_alpha(sprite, pad=trim_pad, alpha_threshold=trim_threshold)
    scale = min(target_w / sprite.width, target_h / sprite.height)
    new_size = (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale)))
    return sprite.resize(new_size, Image.Resampling.LANCZOS)


def fit_template_canvas(
    sprite: Image.Image,
    target_h: int = 184,
    target_w: int = 176,
) -> Image.Image:
    """Scale the full action template canvas into the atlas cell without trimming."""

    scale = min(target_w / sprite.width, target_h / sprite.height)
    new_size = (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale)))
    return sprite.resize(new_size, Image.Resampling.LANCZOS)


def remove_waiting_residue(sprite: Image.Image) -> Image.Image:
    """Drop the detached waiting-state bubble/check residue at the upper right."""

    out = sprite.copy()
    alpha = out.getchannel("A")
    mask = alpha.point(lambda p: 255 if p > 12 else 0, "L")
    w, h = out.size
    cleared = False
    for area, (x0, y0, x1, y1), comp_mask in connected_components(mask):
        if area < 18:
            continue
        near_upper_right = x0 >= round(w * 0.78) and y1 <= round(h * 0.34)
        compact_mark = (x1 - x0) <= round(w * 0.20) and (y1 - y0) <= round(h * 0.16)
        if near_upper_right and compact_mark:
            alpha = ImageChops.subtract(alpha, comp_mask)
            cleared = True
    if cleared:
        out.putalpha(alpha)
    return out


def pad_sprite_canvas(sprite: Image.Image, pad_x: int = 8, pad_y: int = 4) -> Image.Image:
    out = Image.new("RGBA", (sprite.width + pad_x * 2, sprite.height + pad_y * 2), (0, 0, 0, 0))
    out.alpha_composite(sprite, (pad_x, pad_y))
    return out


def remove_review_residue(sprite: Image.Image) -> Image.Image:
    """Drop the thin blue scanline at the far right edge of the review pose."""

    out = sprite.copy()
    px = out.load()
    alpha = out.getchannel("A")
    ap = alpha.load()
    w, h = out.size
    cleared_cols: list[int] = []
    min_hits = max(24, round(h * 0.35))
    for x in range(w - 1, max(-1, w - 20), -1):
        hits = 0
        for y in range(h):
            if ap[x, y] <= 12:
                continue
            r, g, b, _a = px[x, y]
            if b >= g + 10 and b >= r + 24:
                hits += 1
        if hits >= min_hits:
            cleared_cols.append(x)
        elif cleared_cols:
            break
    if not cleared_cols:
        return out
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((min(cleared_cols), 0, max(cleared_cols), h), fill=0)
    out.putalpha(alpha)
    return out


def translate_rgba(image: Image.Image, dx: float, dy: float) -> Image.Image:
    """Translate an RGBA image without wrapping pixels around the edges."""

    dx_i = round(dx)
    dy_i = round(dy)
    if dx_i == 0 and dy_i == 0:
        return image
    w, h = image.size
    src_x0 = max(0, -dx_i)
    src_y0 = max(0, -dy_i)
    src_x1 = min(w, w - dx_i)
    src_y1 = min(h, h - dy_i)
    out = Image.new("RGBA", image.size, (0, 0, 0, 0))
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    out.alpha_composite(image.crop((src_x0, src_y0, src_x1, src_y1)), (max(0, dx_i), max(0, dy_i)))
    return out


def alpha_composite_clipped(base: Image.Image, layer: Image.Image, dest: tuple[int, int]) -> None:
    dx, dy = dest
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(layer.width, base.width - dx)
    src_y1 = min(layer.height, base.height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return
    base.alpha_composite(layer.crop((src_x0, src_y0, src_x1, src_y1)), (max(0, dx), max(0, dy)))


def selected_region_mask(sprite: Image.Image, selector, grow: int = 3, blur: float = 0.45) -> Image.Image:
    w, h = sprite.size
    px = sprite.load()
    mask = Image.new("L", sprite.size, 0)
    mp = mask.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a > 12 and selector(x, y, w, h, r, g, b, a):
                mp[x, y] = 255
    if mask.getbbox() and grow > 1:
        mask = mask.filter(ImageFilter.MaxFilter(grow))
    if mask.getbbox() and blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def shift_selected_region(sprite: Image.Image, selector, dx: float, dy: float) -> Image.Image:
    """Shift a small alpha-masked sprite region for hair, skirt, or prop lag."""

    dx_i = round(dx)
    dy_i = round(dy)
    if dx_i == 0 and dy_i == 0:
        return sprite

    mask = selected_region_mask(sprite, selector)
    if not mask.getbbox():
        return sprite

    transparent = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    selected = Image.composite(sprite, transparent, mask)
    base = Image.composite(transparent, sprite, mask)
    base.alpha_composite(translate_rgba(selected, dx_i, dy_i))
    return base


def transform_selected_region(
    sprite: Image.Image,
    selector,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    rotate: float = 0.0,
    anchor: str = "center",
    grow: int = 3,
    blur: float = 0.45,
) -> Image.Image:
    mask = selected_region_mask(sprite, selector, grow=grow, blur=blur)
    bbox = mask.getbbox()
    if not bbox:
        return sprite

    transparent = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    selected = Image.composite(sprite, transparent, mask)
    base = Image.composite(transparent, sprite, mask)
    x0, y0, x1, y1 = bbox
    crop = selected.crop(bbox)
    new_size = (
        max(1, round(crop.width * scale_x)),
        max(1, round(crop.height * scale_y)),
    )
    transformed = crop.resize(new_size, Image.Resampling.BICUBIC)
    if abs(rotate) > 0.01:
        transformed = transformed.rotate(rotate, expand=True, resample=Image.Resampling.BICUBIC)

    if anchor == "top":
        paste_x = round((x0 + x1) / 2 - transformed.width / 2 + dx)
        paste_y = round(y0 + dy)
    elif anchor == "bottom":
        paste_x = round((x0 + x1) / 2 - transformed.width / 2 + dx)
        paste_y = round(y1 - transformed.height + dy)
    else:
        paste_x = round((x0 + x1) / 2 - transformed.width / 2 + dx)
        paste_y = round((y0 + y1) / 2 - transformed.height / 2 + dy)

    alpha_composite_clipped(base, transformed, (paste_x, paste_y))
    return base


def find_eye_boxes(sprite: Image.Image) -> list[tuple[int, int, int, int]]:
    w, h = sprite.size
    px = sprite.load()
    mask = Image.new("L", sprite.size, 0)
    mp = mask.load()
    for y in range(round(h * 0.20), round(h * 0.56)):
        for x in range(round(w * 0.12), round(w * 0.88)):
            r, g, b, a = px[x, y]
            if a > 80 and r > 55 and b > 55 and g < 118 and (r - g > 22 or b - g > 22):
                mp[x, y] = 255
    comps = connected_components(mask)
    boxes = []
    for area, box, _comp_mask in comps:
        x0, y0, x1, y1 = box
        bw = x1 - x0
        bh = y1 - y0
        if 6 <= area <= 900 and 4 <= bw <= round(w * 0.22) and 4 <= bh <= round(h * 0.18):
            boxes.append(box)
    if boxes:
        boxes = sorted(boxes[:2], key=lambda item: item[0])
    if not boxes:
        eye_w = max(10, round(w * 0.12))
        eye_h = max(12, round(h * 0.10))
        cy = round(h * 0.41)
        boxes = [
            (round(w * 0.36), cy - eye_h // 2, round(w * 0.36) + eye_w, cy + eye_h // 2),
            (round(w * 0.55), cy - eye_h // 2, round(w * 0.55) + eye_w, cy + eye_h // 2),
        ]
    return boxes


def add_eye_overlay(
    sprite: Image.Image,
    blink: float,
    focus: float,
    eye_shift: float,
    brow: float,
    blush: float,
    sparkle: float,
) -> Image.Image:
    if blink <= 0 and focus <= 0 and eye_shift == 0 and brow == 0 and blush <= 0 and sparkle <= 0:
        return sprite

    out = sprite.copy()
    layer = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = sprite.size
    skin = (255, 224, 202, 238)
    line = (54, 42, 45, 230)
    boxes = find_eye_boxes(sprite)

    for x0, y0, x1, y1 in boxes:
        cx = round((x0 + x1) / 2)
        cy = round((y0 + y1) / 2)
        eye_w = max(8, round((x1 - x0) * 0.75))
        eye_h = max(8, round((y1 - y0) * 0.85))
        if blush > 0:
            blush_alpha = max(0, min(100, round(62 * blush)))
            cheek_y = cy + round(eye_h * 1.15)
            draw.ellipse(
                (
                    cx - round(eye_w * 0.8),
                    cheek_y - round(eye_h * 0.24),
                    cx + round(eye_w * 0.8),
                    cheek_y + round(eye_h * 0.24),
                ),
                fill=(255, 139, 169, blush_alpha),
            )
        if blink > 0:
            cover_h = round((eye_h * 0.25) + (eye_h * 1.55 * min(1.0, blink)))
            cover_y0 = cy - eye_h
            cover_y1 = min(cy + eye_h, cover_y0 + cover_h)
            draw.rounded_rectangle((cx - eye_w, cover_y0, cx + eye_w, cover_y1), radius=4, fill=skin)
            if blink >= 0.42:
                line_y = min(cy + 2, cover_y1 - 2)
                arc = max(1, round(2 * abs(eye_shift)))
                draw.line((cx - eye_w + 2, line_y - arc, cx + eye_w - 2, line_y + arc), fill=line, width=2)
        elif eye_shift:
            shift_px = round(eye_shift * max(2, eye_w * 0.35))
            iris_alpha = max(0, min(150, round(95 * abs(eye_shift))))
            draw.ellipse(
                (
                    cx + shift_px - round(eye_w * 0.25),
                    cy - round(eye_h * 0.30),
                    cx + shift_px + round(eye_w * 0.25),
                    cy + round(eye_h * 0.30),
                ),
                fill=(120, 35, 92, iris_alpha),
            )
        if focus > 0:
            alpha = max(0, min(210, round(170 * focus)))
            draw.line(
                (cx - eye_w, cy - eye_h + 2, cx + eye_w, cy - eye_h + 4),
                fill=(43, 36, 38, alpha),
                width=2,
            )
        if brow:
            alpha = max(0, min(210, round(155 * min(1.0, abs(brow)))))
            tilt = round(eye_h * 0.35 * brow)
            by = cy - eye_h - 3
            draw.line((cx - eye_w + 2, by + tilt, cx + eye_w - 2, by - tilt), fill=(62, 47, 41, alpha), width=2)
        if sparkle > 0 and blink < 0.35:
            alpha = max(0, min(220, round(160 * sparkle)))
            sx = cx + round(eye_w * 0.35)
            sy = cy - round(eye_h * 0.25)
            draw.ellipse((sx - 1, sy - 1, sx + 2, sy + 2), fill=(255, 246, 226, alpha))

    out.alpha_composite(layer)
    return out


def add_waving_arm(sprite: Image.Image, raise_amount: float) -> Image.Image:
    if raise_amount <= 0:
        return sprite

    out = sprite.copy()
    layer = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = sprite.size
    t = max(0.0, min(1.0, raise_amount))
    shoulder = (round(w * 0.35), round(h * 0.58))
    elbow = (round(w * (0.30 - 0.05 * t)), round(h * (0.55 - 0.12 * t)))
    hand = (round(w * (0.24 - 0.07 * t)), round(h * (0.53 - 0.22 * t)))
    sleeve_width = max(4, round(w * 0.055))

    draw.line((shoulder, elbow, hand), fill=(43, 37, 39, 235), width=sleeve_width + 2)
    draw.line((shoulder, elbow, hand), fill=(246, 132, 36, 245), width=max(2, sleeve_width - 1))
    hx, hy = hand
    hand_r = max(4, round(w * 0.038))
    draw.ellipse(
        (hx - hand_r, hy - hand_r, hx + hand_r, hy + hand_r),
        fill=(255, 224, 195, 248),
        outline=(53, 42, 45, 225),
        width=1,
    )
    out.alpha_composite(layer)
    return out


def add_hug_overlay(sprite: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return sprite

    out = sprite.copy()
    layer = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = sprite.size
    alpha = max(0, min(220, round(190 * amount)))
    left = (round(w * 0.36), round(h * 0.58), round(w * 0.49), round(h * 0.67))
    right = (round(w * 0.64), round(h * 0.58), round(w * 0.51), round(h * 0.67))
    for start, end in ((left[:2], left[2:]), (right[:2], right[2:])):
        draw.line((start, end), fill=(42, 36, 39, alpha), width=max(3, round(w * 0.04)))
        draw.line((start, end), fill=(255, 245, 231, alpha), width=max(1, round(w * 0.02)))
    out.alpha_composite(layer)
    return out


def add_run_stride_overlay(sprite: Image.Image, stride: float, direction: int) -> Image.Image:
    if abs(stride) <= 0.01:
        return sprite

    out = sprite.copy()
    layer = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = sprite.size
    phase = 1 if stride >= 0 else -1
    strength = min(1.0, abs(stride))
    hip_y = round(h * 0.70)
    knee_y = round(h * (0.80 - 0.03 * strength))
    foot_y = round(h * (0.91 - 0.03 * strength))
    cx = round(w * 0.50)
    front_foot_x = round(cx + direction * phase * w * (0.14 + 0.04 * strength))
    back_foot_x = round(cx - direction * phase * w * (0.12 + 0.03 * strength))
    front_knee_x = round(cx + direction * phase * w * 0.07)
    back_knee_x = round(cx - direction * phase * w * 0.05)
    outline = (43, 35, 38, 220)
    sock = (255, 246, 232, 230)
    accent = (246, 132, 36, 235)
    width = max(3, round(w * 0.035))

    for path in (
        ((cx - direction * phase * 5, hip_y), (front_knee_x, knee_y), (front_foot_x, foot_y)),
        ((cx + direction * phase * 4, hip_y + 2), (back_knee_x, knee_y + 3), (back_foot_x, foot_y + 3)),
    ):
        draw.line(path, fill=outline, width=width + 3, joint="curve")
        draw.line(path, fill=sock, width=max(2, width), joint="curve")

    boot_w = max(8, round(w * 0.085))
    boot_h = max(5, round(h * 0.035))
    for foot_x, foot_y_i in ((front_foot_x, foot_y), (back_foot_x, foot_y + 3)):
        draw.rounded_rectangle(
            (foot_x - boot_w // 2, foot_y_i - boot_h // 2, foot_x + boot_w // 2, foot_y_i + boot_h // 2),
            radius=2,
            fill=outline,
        )
        draw.line(
            (foot_x - boot_w // 4, foot_y_i - 1, foot_x + boot_w // 4, foot_y_i - 1),
            fill=accent,
            width=1,
        )

    out.alpha_composite(layer)
    return out


def add_sitting_legs_overlay(sprite: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return sprite

    out = sprite.copy()
    layer = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = sprite.size
    alpha = max(0, min(235, round(220 * amount)))
    skirt_y = round(h * 0.70)
    knee_y = round(h * 0.82)
    foot_y = round(h * 0.90)
    cx = round(w * 0.50)
    outline = (42, 35, 38, alpha)
    sock = (255, 246, 232, alpha)
    accent = (247, 132, 36, alpha)
    width = max(3, round(w * 0.04))
    paths = (
        ((cx - 8, skirt_y), (round(w * 0.36), knee_y), (round(w * 0.25), foot_y)),
        ((cx + 8, skirt_y), (round(w * 0.64), knee_y), (round(w * 0.75), foot_y)),
    )
    for path in paths:
        draw.line(path, fill=outline, width=width + 3, joint="curve")
        draw.line(path, fill=sock, width=width, joint="curve")
    boot_w = max(10, round(w * 0.09))
    boot_h = max(6, round(h * 0.04))
    for foot_x in (round(w * 0.25), round(w * 0.75)):
        draw.rounded_rectangle(
            (foot_x - boot_w // 2, foot_y - boot_h // 2, foot_x + boot_w // 2, foot_y + boot_h // 2),
            radius=2,
            fill=outline,
        )
        draw.line((foot_x - boot_w // 4, foot_y - 1, foot_x + boot_w // 4, foot_y - 1), fill=accent, width=1)
    out.alpha_composite(layer)
    return out


def apply_crouch_pose(sprite: Image.Image, amount: float) -> Image.Image:
    amount = max(0.0, min(1.0, amount))
    if amount <= 0:
        return sprite

    out = transform_selected_region(
        sprite,
        lambda x, y, w, h, *_: y > h * 0.58 and w * 0.18 < x < w * 0.82,
        scale_x=1.0 + 0.06 * amount,
        scale_y=1.0 - 0.24 * amount,
        dy=round(2 * amount),
        anchor="bottom",
        grow=5,
        blur=0.55,
    )
    out = transform_selected_region(
        out,
        lambda x, y, w, h, *_: y < h * 0.68 or ((x < w * 0.30 or x > w * 0.70) and y < h * 0.84),
        scale_x=1.0 + 0.012 * amount,
        scale_y=1.0 - 0.025 * amount,
        dy=round(5 * amount),
        anchor="top",
        grow=3,
        blur=0.35,
    )
    return out


def apply_sit_pose(sprite: Image.Image, amount: float) -> Image.Image:
    amount = max(0.0, min(1.0, amount))
    if amount <= 0:
        return sprite

    out = transform_selected_region(
        sprite,
        lambda x, y, w, h, *_: y > h * 0.55 and w * 0.18 < x < w * 0.82,
        scale_x=1.0 + 0.16 * amount,
        scale_y=1.0 - 0.42 * amount,
        dy=round(7 * amount),
        anchor="bottom",
        grow=5,
        blur=0.65,
    )
    out = transform_selected_region(
        out,
        lambda x, y, w, h, *_: y < h * 0.66 or ((x < w * 0.30 or x > w * 0.70) and y < h * 0.88),
        scale_x=1.0 + 0.018 * amount,
        scale_y=1.0 - 0.035 * amount,
        dy=round(12 * amount),
        anchor="top",
        grow=3,
        blur=0.35,
    )
    out = add_sitting_legs_overlay(out, amount)
    return out


def apply_run_stride(sprite: Image.Image, stride: float, direction: int) -> Image.Image:
    if abs(stride) <= 0.01:
        return sprite

    phase = 1 if stride >= 0 else -1
    strength = min(1.0, abs(stride))
    out = shift_selected_region(
        sprite,
        lambda x, y, w, h, *_: y > h * 0.70 and w * 0.28 < x < w * 0.50,
        direction * phase * (4 + 2 * strength),
        -2 * strength,
    )
    out = shift_selected_region(
        out,
        lambda x, y, w, h, *_: y > h * 0.70 and w * 0.50 <= x < w * 0.72,
        -direction * phase * (3 + 2 * strength),
        2 * strength,
    )
    out = add_run_stride_overlay(out, stride, direction)
    return out
def eased_band(value: float, start: float, peak: float, end: float) -> float:
    if value <= start or value >= end:
        return 0.0
    if value <= peak:
        t = (value - start) / max(1e-6, peak - start)
    else:
        t = (end - value) / max(1e-6, end - peak)
    t = clampf(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def live2d_source_point(x_rel: float, y_rel: float, w: int, h: int, motion: Motion) -> tuple[float, float]:
    x = x_rel * w
    y = y_rel * h
    cx = w * 0.5
    head_cx = w * 0.5
    head_cy = h * 0.28
    torso_cy = h * 0.54

    head = eased_band(y_rel, 0.00, 0.16, 0.48)
    torso = eased_band(y_rel, 0.18, 0.46, 0.84)
    skirt = eased_band(y_rel, 0.58, 0.82, 1.00)
    upper = eased_band(y_rel, 0.10, 0.30, 0.60)

    if torso:
        torso_scale_x = 1.0 + 0.020 * motion.torso_breathe * torso
        torso_scale_y = 1.0 - 0.015 * motion.torso_breathe * torso
        x = cx + (x - cx) / max(0.88, torso_scale_x)
        y = torso_cy + (y - torso_cy) / max(0.88, torso_scale_y)

    if head:
        head_scale_x = 1.0 - 0.030 * abs(motion.head_turn) * head
        head_scale_y = 1.0 + 0.018 * motion.head_nod * head
        x = head_cx + (x - head_cx) / max(0.88, head_scale_x)
        y = head_cy + (y - head_cy) / max(0.88, head_scale_y)

    left_hair = clampf((0.34 - x_rel) / 0.20, 0.0, 1.0) * eased_band(y_rel, 0.16, 0.50, 1.00)
    right_hair = clampf((x_rel - 0.66) / 0.20, 0.0, 1.0) * eased_band(y_rel, 0.16, 0.50, 1.00)
    left_skirt = clampf((0.50 - x_rel) / 0.32, 0.0, 1.0)
    right_skirt = clampf((x_rel - 0.50) / 0.32, 0.0, 1.0)

    dx = 0.0
    dy = 0.0
    dx += motion.head_turn * head * 5.2
    dy -= motion.head_nod * head * 4.0
    dx += motion.torso_lean * torso * 4.2
    dy -= motion.torso_breathe * torso * 1.4
    dx += motion.hair_sway * upper * 2.2
    dx += motion.left_hair_sway * left_hair * 4.4
    dx += motion.right_hair_sway * right_hair * 4.4
    dy += abs(motion.left_hair_sway) * left_hair * 1.2
    dy += abs(motion.right_hair_sway) * right_hair * 1.2
    dx += motion.skirt_sway * skirt * 2.0
    dx += (-motion.skirt_fan * left_skirt + motion.skirt_fan * right_skirt) * skirt * 3.6
    dy += (motion.skirt_lift - abs(motion.skirt_fan) * 0.35) * skirt * 1.8

    src_x = clampf(x - dx, 0.0, w - 1.0)
    src_y = clampf(y - dy, 0.0, h - 1.0)
    return (src_x, src_y)


def apply_live2d_style_pose(sprite: Image.Image, motion: Motion) -> Image.Image:
    """Warp the full sprite canvas with a small mesh, without any cutout motion."""

    has_warp = any(
        abs(value) > 0.001
        for value in (
            motion.hair_sway,
            motion.skirt_sway,
            motion.skirt_lift,
            motion.head_turn,
            motion.head_nod,
            motion.torso_breathe,
            motion.torso_lean,
            motion.left_hair_sway,
            motion.right_hair_sway,
            motion.skirt_fan,
        )
    )
    if not has_warp:
        return sprite

    w, h = sprite.size
    grid_x = (0.0, 0.25, 0.50, 0.75, 1.0)
    grid_y = (0.0, 0.16, 0.32, 0.52, 0.72, 0.88, 1.0)
    mesh = []
    for yi in range(len(grid_y) - 1):
        for xi in range(len(grid_x) - 1):
            x0_rel = grid_x[xi]
            x1_rel = grid_x[xi + 1]
            y0_rel = grid_y[yi]
            y1_rel = grid_y[yi + 1]
            x0 = round(x0_rel * w)
            x1 = round(x1_rel * w)
            y0 = round(y0_rel * h)
            y1 = round(y1_rel * h)
            if x1 <= x0 or y1 <= y0:
                continue
            quad = []
            for px_rel, py_rel in (
                (x0_rel, y0_rel),
                (x0_rel, y1_rel),
                (x1_rel, y1_rel),
                (x1_rel, y0_rel),
            ):
                sx, sy = live2d_source_point(px_rel, py_rel, w, h, motion)
                quad.extend((sx, sy))
            mesh.append(((x0, y0, x1, y1), tuple(quad)))
    return sprite.transform(sprite.size, Image.Transform.MESH, mesh, resample=Image.Resampling.BICUBIC)


def apply_local_motion(sprite: Image.Image, motion: Motion) -> Image.Image:
    out = apply_live2d_style_pose(sprite.copy(), motion)
    out = add_eye_overlay(
        out,
        0.0,
        motion.focus,
        motion.eye_shift,
        motion.brow,
        motion.blush,
        motion.sparkle,
    )
    out = add_waving_arm(out, motion.arm_raise)
    return out


def transform_sprite(sprite: Image.Image, motion: Motion) -> Image.Image:
    posed = apply_local_motion(sprite, motion)
    w, h = posed.size
    scaled = posed.resize(
        (max(1, round(w * motion.scale_x)), max(1, round(h * motion.scale_y))),
        Image.Resampling.BICUBIC,
    )
    if abs(motion.rotate) > 0.01:
        scaled = scaled.rotate(motion.rotate, expand=True, resample=Image.Resampling.BICUBIC)
    if motion.opacity < 1.0:
        alpha = scaled.getchannel("A").point(lambda p: round(p * motion.opacity))
        scaled.putalpha(alpha)
    return scaled


def add_cube_glow(cell: Image.Image, strength: float, center: tuple[int, int]) -> None:
    if strength <= 0:
        return
    # Subtle attached glow around the held cube area, intentionally not floating.
    layer = Image.new("RGBA", cell.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = max(0, min(64, round(42 * strength)))
    cx, cy = center
    draw.ellipse((cx - 34, cy - 28, cx + 34, cy + 28), fill=(255, 178, 55, alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(7))
    cell.alpha_composite(layer)


def add_cube_activity(cell: Image.Image, motion: Motion, center: tuple[int, int]) -> None:
    """Draw a tiny attached diamond cue for Codex-working cube activity."""

    if motion.cube_spin <= 0:
        return
    cx, cy = center
    alpha = max(0, min(210, round(115 + 90 * min(1.0, motion.glow))))
    angle = motion.cube_spin * math.tau
    rx = 8
    ry = 6
    points = []
    for idx in range(4):
        a = angle + math.pi / 4 + idx * math.pi / 2
        points.append((round(cx + math.cos(a) * rx), round(cy + math.sin(a) * ry)))
    draw = ImageDraw.Draw(cell)
    draw.line(points + [points[0]], fill=(255, 237, 156, alpha), width=2)
    draw.line(
        (
            round(cx - math.cos(angle) * 5),
            round(cy - math.sin(angle) * 4),
            round(cx + math.cos(angle) * 5),
            round(cy + math.sin(angle) * 4),
        ),
        fill=(255, 255, 226, alpha),
        width=1,
    )


def make_cell(sprite: Image.Image, motion: Motion, baseline: int = 198) -> Image.Image:
    cell = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
    moved = transform_sprite(sprite, motion)
    x = (CELL_W - moved.width) // 2 + motion.x
    y = baseline - moved.height + motion.y
    cube_center = (
        x + round(moved.width * 0.50) + round(motion.cube_dx * 0.25),
        y + round(moved.height * 0.58) + round(motion.cube_dy * 0.25),
    )
    add_cube_glow(cell, motion.glow, cube_center)
    cell.alpha_composite(moved, (x, y))
    add_cube_activity(cell, motion, cube_center)
    return cell


def state_motions(state: str, frame_count: int) -> list[Motion]:
    """Return low-motion, state-specific frame transforms."""

    if state == "idle":
        motions = [
            Motion(glow=0.14, blush=0.08, torso_breathe=0.16, left_hair_sway=-0.10, right_hair_sway=0.10),
            Motion(y=-1, scale_y=1.008, glow=0.15, blush=0.08, sparkle=0.02, torso_breathe=0.34, left_hair_sway=-0.16, right_hair_sway=0.14),
            Motion(y=-3, scale_y=1.016, glow=0.16, blush=0.08, sparkle=0.03, torso_breathe=0.58, left_hair_sway=-0.24, right_hair_sway=0.22),
            Motion(y=-5, scale_y=1.026, glow=0.18, blush=0.08, sparkle=0.05, torso_breathe=0.94, left_hair_sway=-0.34, right_hair_sway=0.32),
            Motion(y=-5, scale_y=1.026, glow=0.18, blush=0.08, sparkle=0.05, torso_breathe=0.90, left_hair_sway=-0.30, right_hair_sway=0.28),
            Motion(y=-3, scale_y=1.016, glow=0.16, blush=0.08, sparkle=0.03, torso_breathe=0.54, left_hair_sway=-0.18, right_hair_sway=0.16),
            Motion(y=-1, scale_y=1.008, glow=0.15, blush=0.08, torso_breathe=0.26, left_hair_sway=0.08, right_hair_sway=-0.08),
            Motion(glow=0.14, blush=0.08, torso_breathe=0.12, left_hair_sway=0.02, right_hair_sway=-0.02),
        ]
    elif state == "running-right":
        motions = [
            Motion(x=-2, y=3, scale_x=1.012, scale_y=0.982, rotate=-1.1, hair_sway=-2, skirt_sway=-1, glow=0.16, focus=0.10, eye_shift=0.10, torso_lean=-0.36, torso_breathe=0.24, head_turn=-0.24, left_hair_sway=-0.60, right_hair_sway=-0.18, skirt_fan=0.12),
            Motion(x=0, y=-1, scale_y=1.006, rotate=-1.8, hair_sway=-3, skirt_sway=-1, skirt_lift=-1, glow=0.17, focus=0.10, eye_shift=0.12, torso_lean=-0.58, torso_breathe=0.28, head_turn=-0.34, head_nod=0.08, left_hair_sway=-0.86, right_hair_sway=-0.26, skirt_fan=0.18),
            Motion(x=4, y=-7, scale_x=0.996, scale_y=1.018, rotate=-2.4, hair_sway=-4, skirt_sway=-2, skirt_lift=-2, glow=0.18, focus=0.12, eye_shift=0.14, cube_dy=-1, torso_lean=-0.80, torso_breathe=0.34, head_turn=-0.40, head_nod=0.14, left_hair_sway=-1.08, right_hair_sway=-0.34, skirt_fan=0.24),
            Motion(x=7, y=-3, scale_y=1.010, rotate=-1.9, hair_sway=-2, skirt_sway=-1, skirt_lift=-1, glow=0.18, focus=0.12, eye_shift=0.12, torso_lean=-0.60, torso_breathe=0.26, head_turn=-0.24, head_nod=0.10, left_hair_sway=-0.72, right_hair_sway=-0.18, skirt_fan=0.18),
            Motion(x=5, y=2, scale_x=1.010, scale_y=0.986, rotate=-1.3, hair_sway=-2, skirt_sway=-1, glow=0.16, focus=0.10, eye_shift=0.10, torso_lean=0.18, torso_breathe=0.24, head_turn=0.08, left_hair_sway=0.20, right_hair_sway=0.52, skirt_fan=0.12),
            Motion(x=2, y=-1, scale_y=1.006, rotate=-1.8, hair_sway=-3, skirt_sway=-1, skirt_lift=-1, glow=0.17, focus=0.10, eye_shift=0.12, torso_lean=0.40, torso_breathe=0.28, head_turn=0.22, head_nod=0.06, left_hair_sway=0.26, right_hair_sway=0.78, skirt_fan=0.18),
            Motion(x=-1, y=-7, scale_x=0.996, scale_y=1.018, rotate=-2.4, hair_sway=-4, skirt_sway=-2, skirt_lift=-2, glow=0.18, focus=0.12, eye_shift=0.14, cube_dy=-1, torso_lean=0.62, torso_breathe=0.34, head_turn=0.34, head_nod=0.12, left_hair_sway=0.34, right_hair_sway=1.02, skirt_fan=0.24),
            Motion(x=-3, y=-3, scale_y=1.010, rotate=-1.9, hair_sway=-2, skirt_sway=-1, skirt_lift=-1, glow=0.17, focus=0.10, eye_shift=0.12, torso_lean=0.32, torso_breathe=0.28, head_turn=0.18, head_nod=0.08, left_hair_sway=0.18, right_hair_sway=0.62, skirt_fan=0.18),
        ]
    elif state == "running-left":
        motions = [
            Motion(x=2, y=3, scale_x=1.012, scale_y=0.982, rotate=1.1, hair_sway=2, skirt_sway=1, glow=0.16, focus=0.10, eye_shift=-0.10, torso_lean=0.36, torso_breathe=0.24, head_turn=0.24, left_hair_sway=0.18, right_hair_sway=0.60, skirt_fan=0.12),
            Motion(x=0, y=-1, scale_y=1.006, rotate=1.8, hair_sway=3, skirt_sway=1, skirt_lift=-1, glow=0.17, focus=0.10, eye_shift=-0.12, torso_lean=0.58, torso_breathe=0.28, head_turn=0.34, head_nod=0.08, left_hair_sway=0.26, right_hair_sway=0.86, skirt_fan=0.18),
            Motion(x=-4, y=-7, scale_x=0.996, scale_y=1.018, rotate=2.4, hair_sway=4, skirt_sway=2, skirt_lift=-2, glow=0.18, focus=0.12, eye_shift=-0.14, cube_dy=-1, torso_lean=0.80, torso_breathe=0.34, head_turn=0.40, head_nod=0.14, left_hair_sway=0.34, right_hair_sway=1.08, skirt_fan=0.24),
            Motion(x=-7, y=-3, scale_y=1.010, rotate=1.9, hair_sway=2, skirt_sway=1, skirt_lift=-1, glow=0.18, focus=0.12, eye_shift=-0.12, torso_lean=0.60, torso_breathe=0.26, head_turn=0.24, head_nod=0.10, left_hair_sway=0.18, right_hair_sway=0.72, skirt_fan=0.18),
            Motion(x=-5, y=2, scale_x=1.010, scale_y=0.986, rotate=1.3, hair_sway=2, skirt_sway=1, glow=0.16, focus=0.10, eye_shift=-0.10, torso_lean=-0.18, torso_breathe=0.24, head_turn=-0.08, left_hair_sway=-0.52, right_hair_sway=-0.20, skirt_fan=0.12),
            Motion(x=-2, y=-1, scale_y=1.006, rotate=1.8, hair_sway=3, skirt_sway=1, skirt_lift=-1, glow=0.17, focus=0.10, eye_shift=-0.12, torso_lean=-0.40, torso_breathe=0.28, head_turn=-0.22, head_nod=0.06, left_hair_sway=-0.78, right_hair_sway=-0.26, skirt_fan=0.18),
            Motion(x=1, y=-7, scale_x=0.996, scale_y=1.018, rotate=2.4, hair_sway=4, skirt_sway=2, skirt_lift=-2, glow=0.18, focus=0.12, eye_shift=-0.14, cube_dy=-1, torso_lean=-0.62, torso_breathe=0.34, head_turn=-0.34, head_nod=0.12, left_hair_sway=-1.02, right_hair_sway=-0.34, skirt_fan=0.24),
            Motion(x=3, y=-3, scale_y=1.010, rotate=1.9, hair_sway=2, skirt_sway=1, skirt_lift=-1, glow=0.17, focus=0.10, eye_shift=-0.12, torso_lean=-0.32, torso_breathe=0.28, head_turn=-0.18, head_nod=0.08, left_hair_sway=-0.62, right_hair_sway=-0.18, skirt_fan=0.18),
        ]
    elif state == "waving":
        motions = [
            Motion(rotate=-0.5, glow=0.18, blush=0.10, arm_raise=0.28, torso_lean=-0.08, torso_breathe=0.16, head_turn=-0.06, left_hair_sway=-0.18, right_hair_sway=0.22),
            Motion(y=-2, rotate=-1.2, hair_sway=1, glow=0.20, blush=0.12, sparkle=0.06, arm_raise=0.72, torso_lean=-0.20, torso_breathe=0.26, head_turn=-0.12, head_nod=0.06, left_hair_sway=-0.48, right_hair_sway=0.28, skirt_fan=0.12),
            Motion(y=-3, rotate=-1.8, hair_sway=1, glow=0.22, blush=0.14, sparkle=0.10, arm_raise=1.00, torso_lean=-0.34, torso_breathe=0.34, head_turn=-0.20, head_nod=0.12, left_hair_sway=-0.74, right_hair_sway=0.34, skirt_fan=0.18),
            Motion(y=-1, rotate=-0.8, glow=0.19, blush=0.10, arm_raise=0.58, torso_lean=-0.12, torso_breathe=0.20, head_turn=-0.08, left_hair_sway=-0.24, right_hair_sway=0.20),
        ]
    elif state == "jumping":
        motions = [
            Motion(y=4, scale_x=1.028, scale_y=0.944, crouch=0.44, glow=0.18, hair_sway=0, skirt_lift=1, cube_dy=1, torso_breathe=0.18, head_nod=-0.10, left_hair_sway=-0.12, right_hair_sway=0.12, skirt_fan=0.10),
            Motion(y=-4, scale_y=1.018, crouch=0.10, glow=0.22, hair_sway=1, skirt_lift=-1, cube_dy=-1, cube_spin=0.20, sparkle=0.06, torso_breathe=0.30, head_nod=0.16, left_hair_sway=-0.42, right_hair_sway=0.42, skirt_fan=0.22),
            Motion(y=-10, scale_x=0.992, scale_y=1.040, glow=0.26, hair_sway=2, skirt_lift=-3, cube_dy=-3, cube_spin=0.48, sparkle=0.14, torso_breathe=0.40, head_nod=0.26, left_hair_sway=-0.86, right_hair_sway=0.86, skirt_fan=0.34),
            Motion(y=-4, scale_y=1.018, glow=0.22, hair_sway=1, skirt_lift=-1, cube_dy=-1, cube_spin=0.72, sparkle=0.08, torso_breathe=0.28, head_nod=0.14, left_hair_sway=-0.48, right_hair_sway=0.48, skirt_fan=0.20),
            Motion(y=3, scale_x=1.020, scale_y=0.958, crouch=0.34, glow=0.18, hair_sway=-1, skirt_lift=1, cube_dy=1, cube_spin=0.92, torso_breathe=0.20, head_nod=-0.08, left_hair_sway=0.22, right_hair_sway=-0.22, skirt_fan=0.10),
        ]
    elif state == "failed":
        motions = [
            Motion(y=2, scale_y=0.996, rotate=0.0, glow=0.07, hug=0.18, brow=0.28, torso_breathe=0.14, head_nod=-0.18, torso_lean=-0.10, left_hair_sway=0.08, right_hair_sway=-0.08),
            Motion(y=3, scale_x=1.004, scale_y=0.992, rotate=-0.3, glow=0.07, hug=0.24, brow=0.40, torso_breathe=0.18, head_nod=-0.24, torso_lean=-0.16, left_hair_sway=0.12, right_hair_sway=-0.12),
            Motion(y=4, scale_x=1.006, scale_y=0.988, rotate=-0.7, glow=0.06, hug=0.30, brow=0.56, torso_breathe=0.22, head_nod=-0.30, torso_lean=-0.22, left_hair_sway=0.18, right_hair_sway=-0.18),
            Motion(y=3, scale_x=1.002, scale_y=0.992, rotate=-0.4, glow=0.06, hug=0.24, brow=0.40, torso_breathe=0.16, head_nod=-0.22, torso_lean=-0.14, left_hair_sway=0.10, right_hair_sway=-0.10),
            Motion(y=2, scale_y=0.996, rotate=0.0, glow=0.07, hug=0.18, brow=0.28, torso_breathe=0.12, head_nod=-0.16, torso_lean=-0.08, left_hair_sway=0.06, right_hair_sway=-0.06),
            Motion(y=3, scale_x=1.004, scale_y=0.992, rotate=-0.2, glow=0.07, hug=0.22, brow=0.36, torso_breathe=0.16, head_nod=-0.22, torso_lean=-0.12, left_hair_sway=0.08, right_hair_sway=-0.08),
            Motion(y=4, scale_x=1.006, scale_y=0.988, rotate=-0.5, glow=0.06, hug=0.28, brow=0.50, torso_breathe=0.20, head_nod=-0.28, torso_lean=-0.18, left_hair_sway=0.14, right_hair_sway=-0.14),
            Motion(y=3, scale_x=1.002, scale_y=0.992, rotate=-0.2, glow=0.07, hug=0.20, brow=0.32, torso_breathe=0.15, head_nod=-0.20, torso_lean=-0.10, left_hair_sway=0.08, right_hair_sway=-0.08),
        ]
    elif state == "waiting":
        motions = [
            Motion(glow=0.06, blush=0.06, eye_shift=-0.02, torso_breathe=0.10, left_hair_sway=-0.06, right_hair_sway=0.06),
            Motion(y=-1, scale_y=1.006, glow=0.07, blush=0.06, sparkle=0.02, eye_shift=0.00, torso_breathe=0.24, left_hair_sway=-0.12, right_hair_sway=0.10),
            Motion(y=-2, scale_y=1.010, glow=0.08, blush=0.06, sparkle=0.04, eye_shift=0.02, torso_breathe=0.36, left_hair_sway=-0.16, right_hair_sway=0.14),
            Motion(y=-1, scale_y=1.008, glow=0.09, blush=0.06, sparkle=0.05, eye_shift=0.04, torso_breathe=0.40, left_hair_sway=-0.18, right_hair_sway=0.16),
            Motion(scale_y=1.004, glow=0.08, blush=0.06, sparkle=0.03, eye_shift=0.02, torso_breathe=0.22, left_hair_sway=-0.08, right_hair_sway=0.08),
            Motion(glow=0.07, blush=0.06, eye_shift=0.00, torso_breathe=0.12, left_hair_sway=0.04, right_hair_sway=-0.04),
        ]
    elif state == "running":
        motions = [
            Motion(glow=0.28, focus=0.42, eye_shift=-0.04, brow=0.06, sparkle=0.10, cube_spin=0.12, torso_breathe=0.18, head_turn=-0.06, left_hair_sway=-0.14, right_hair_sway=0.12),
            Motion(y=-1, scale_y=1.010, glow=0.34, focus=0.54, eye_shift=0.02, brow=0.10, sparkle=0.18, hair_sway=1, cube_dx=1, cube_dy=-1, cube_spin=0.28, torso_lean=-0.16, torso_breathe=0.26, head_turn=-0.02, head_nod=0.06, left_hair_sway=-0.24, right_hair_sway=0.18),
            Motion(scale_y=1.008, glow=0.40, focus=0.62, eye_shift=0.08, brow=0.14, sparkle=0.28, cube_dx=1, cube_spin=0.46, torso_lean=-0.08, torso_breathe=0.32, head_turn=0.10, head_nod=0.10, left_hair_sway=-0.10, right_hair_sway=0.30, skirt_fan=0.10),
            Motion(y=-1, scale_y=1.012, glow=0.42, focus=0.66, eye_shift=0.10, brow=0.16, sparkle=0.30, hair_sway=1, cube_dx=1, cube_dy=0, cube_spin=0.56, torso_lean=0.02, torso_breathe=0.34, head_turn=0.12, head_nod=0.10, left_hair_sway=0.02, right_hair_sway=0.24, skirt_fan=0.12),
            Motion(y=-1, scale_y=1.010, glow=0.36, focus=0.58, eye_shift=0.02, brow=0.12, sparkle=0.18, hair_sway=-1, cube_dy=1, cube_spin=0.64, torso_lean=0.10, torso_breathe=0.28, head_turn=0.04, head_nod=0.08, left_hair_sway=0.18, right_hair_sway=-0.22, skirt_fan=0.08),
            Motion(scale_y=1.006, glow=0.30, focus=0.48, eye_shift=-0.06, brow=0.08, sparkle=0.12, cube_dx=-1, cube_spin=0.80, torso_lean=0.14, torso_breathe=0.22, head_turn=-0.08, left_hair_sway=0.22, right_hair_sway=-0.16),
            Motion(y=1, scale_y=1.004, glow=0.29, focus=0.44, eye_shift=-0.08, brow=0.07, sparkle=0.10, hair_sway=-1, cube_dx=-1, cube_dy=-1, cube_spin=0.88, torso_lean=0.10, torso_breathe=0.20, head_turn=-0.12, head_nod=-0.02, left_hair_sway=0.16, right_hair_sway=-0.18),
            Motion(glow=0.28, focus=0.40, eye_shift=-0.10, brow=0.06, sparkle=0.10, cube_dy=-1, cube_spin=0.96, torso_lean=0.04, torso_breathe=0.18, head_turn=-0.12, left_hair_sway=0.10, right_hair_sway=-0.12),
        ]
    elif state == "review":
        motions = [
            Motion(scale_y=1.004, rotate=-0.6, glow=0.20, focus=0.62, eye_shift=-0.10, brow=0.26, torso_lean=-0.14, torso_breathe=0.14, head_turn=-0.08, head_nod=0.02, left_hair_sway=-0.12, right_hair_sway=0.10),
            Motion(y=-1, scale_y=1.010, rotate=-1.2, glow=0.23, focus=0.78, eye_shift=-0.16, brow=0.36, torso_lean=-0.24, torso_breathe=0.22, head_turn=-0.16, head_nod=0.06, left_hair_sway=-0.20, right_hair_sway=0.14),
            Motion(y=-2, scale_y=1.014, rotate=-1.6, glow=0.25, focus=0.90, eye_shift=-0.12, brow=0.44, sparkle=0.03, torso_lean=-0.30, torso_breathe=0.30, head_turn=-0.12, head_nod=0.10, left_hair_sway=-0.26, right_hair_sway=0.18),
            Motion(y=-1, scale_y=1.012, rotate=-1.0, glow=0.23, focus=0.82, eye_shift=-0.04, brow=0.40, torso_lean=-0.12, torso_breathe=0.24, head_turn=0.02, head_nod=0.08, left_hair_sway=-0.10, right_hair_sway=0.12),
            Motion(scale_y=1.008, rotate=-0.4, glow=0.21, focus=0.68, eye_shift=0.04, brow=0.28, torso_lean=0.10, torso_breathe=0.18, head_turn=0.10, head_nod=0.04, left_hair_sway=0.06, right_hair_sway=-0.08),
            Motion(scale_y=1.004, rotate=-0.3, glow=0.19, focus=0.58, eye_shift=-0.02, brow=0.24, torso_breathe=0.14, head_turn=0.00, left_hair_sway=-0.04, right_hair_sway=0.04),
        ]
    else:
        motions = [Motion() for _ in range(frame_count)]

    if len(motions) < frame_count:
        motions.extend([motions[-1]] * (frame_count - len(motions)))
    return motions[:frame_count]


def load_pose_sprites() -> dict[str, Image.Image]:
    raw_sprites: dict[str, Image.Image] = {}
    for state, _frame_count in ROWS:
        source_name = POSE_SOURCE_FILES.get(state)
        if source_name:
            source = load_rgb(source_name)
            if state == "running-left":
                source = ImageOps.mirror(source)
            box = POSE_SOURCE_BOXES.get(state)
            if box:
                source = source.crop(box)
            raw_sprite = apply_background_alpha(
                source,
                grow=POSE_MASK_GROW.get(state, 5),
                blur=POSE_MASK_BLUR.get(state, 0.35),
            )
            if state == "waiting":
                raw_sprite = remove_waiting_residue(raw_sprite)
            if state == "review":
                raw_sprite = remove_review_residue(raw_sprite)
            raw_sprites[state] = raw_sprite
        else:
            raise RuntimeError(f"missing pose source for state: {state}")

    target_stats = pose_color_stats(raw_sprites["idle"])
    poses: dict[str, Image.Image] = {}
    for state, raw_sprite in raw_sprites.items():
        normalized = normalize_pose_colors(raw_sprite, target_stats)
        normalized = tune_pose_palette(normalized, state)
        pose = fit_template_canvas(
            normalized,
            target_h=POSE_TARGET_H.get(state, 154),
            target_w=POSE_TARGET_W.get(state, 136),
        )
        normalized.save(POSES_DIR / f"{state}-raw.png")
        pose.save(POSES_DIR / f"{state}.png")
        poses[state] = pose
    return poses


def make_pose_sheet(pose_sprites: dict[str, Image.Image]) -> Image.Image:
    cell_w = 190
    cell_h = 236
    sheet = Image.new("RGBA", (cell_w * 3, cell_h * 3), (255, 249, 240, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (state, _frame_count) in enumerate(ROWS):
        sprite = pose_sprites[state]
        x = (idx % 3) * cell_w + (cell_w - sprite.width) // 2
        y = (idx // 3) * cell_h + 10
        sheet.alpha_composite(sprite, (x, y))
        draw.text(((idx % 3) * cell_w + 8, (idx // 3) * cell_h + cell_h - 24), state, fill=(47, 42, 46, 255))
    return sheet


def make_frames(pose_sprites: dict[str, Image.Image]) -> dict[str, list[Image.Image]]:
    frames: dict[str, list[Image.Image]] = {}
    for state, frame_count in ROWS:
        source = pose_sprites[state]
        frames[state] = [make_cell(source, motion) for motion in state_motions(state, frame_count)]
    return frames


def compose_atlas(frames: dict[str, list[Image.Image]]) -> Image.Image:
    atlas = Image.new("RGBA", (CELL_W * COLUMNS, CELL_H * len(ROWS)), (0, 0, 0, 0))
    for row_idx, (state, _count) in enumerate(ROWS):
        for col_idx, frame in enumerate(frames[state]):
            atlas.alpha_composite(frame, (col_idx * CELL_W, row_idx * CELL_H))
    return normalize_transparent_rgb(atlas)


def normalize_transparent_rgb(image: Image.Image) -> Image.Image:
    """Clear hidden RGB values where alpha is zero.

    Codex pet validation checks transparent-pixel residue because hidden color
    can cause halos after format conversion or rendering.
    """

    rgba = image.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if a == 0 and (r or g or b):
                pixels[x, y] = (0, 0, 0, 0)
    return rgba


def save_frames(frames: dict[str, list[Image.Image]]) -> None:
    for state, state_frames in frames.items():
        state_dir = FRAMES_DIR / state
        state_dir.mkdir(parents=True, exist_ok=True)
        for old_png in state_dir.glob("*.png"):
            old_png.unlink()
        for idx, frame in enumerate(state_frames):
            normalize_transparent_rgb(frame).save(state_dir / f"{idx:02d}.png")


def make_contact_sheet(atlas: Image.Image) -> Image.Image:
    scale = 0.5
    thumb = atlas.resize((round(atlas.width * scale), round(atlas.height * scale)), Image.Resampling.NEAREST)
    sheet = Image.new("RGBA", (thumb.width + 120, thumb.height + 22), (255, 249, 240, 255))
    sheet.alpha_composite(thumb, (106, 22))
    draw = ImageDraw.Draw(sheet)
    for row_idx, (state, frame_count) in enumerate(ROWS):
        y = 22 + row_idx * round(CELL_H * scale) + 8
        draw.text((8, y), f"{row_idx} {state} ({frame_count})", fill=(47, 42, 46, 255))
    for col in range(COLUMNS + 1):
        x = 106 + col * round(CELL_W * scale)
        draw.line((x, 22, x, 22 + thumb.height), fill=(221, 171, 118, 255))
    for row in range(len(ROWS) + 1):
        y = 22 + row * round(CELL_H * scale)
        draw.line((106, y, 106 + thumb.width, y), fill=(221, 171, 118, 255))
    return sheet


def checkerboard(size: tuple[int, int], block: int = 10) -> Image.Image:
    board = Image.new("RGBA", size, (244, 64, 150, 255))
    draw = ImageDraw.Draw(board)
    w, h = size
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=(45, 198, 198, 255))
    return board


def labeled_panel(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGBA")
    panel = Image.new("RGBA", (image.width + 20, image.height + 36), (255, 249, 240, 255))
    panel.alpha_composite(image, (10, 10))
    draw = ImageDraw.Draw(panel)
    draw.text((10, image.height + 16), label, fill=(47, 42, 46, 255))
    return panel


def make_edge_audit_sheet(pose_sprites: dict[str, Image.Image]) -> Image.Image:
    cell_w = 202
    cell_h = 252
    sheet = Image.new("RGBA", (cell_w * 3, cell_h * 3), (255, 249, 240, 255))
    draw = ImageDraw.Draw(sheet)
    for idx, (state, _frame_count) in enumerate(ROWS):
        sprite = pose_sprites[state].convert("RGBA")
        bg = checkerboard((cell_w - 24, cell_h - 56), block=10)
        bx = (bg.width - sprite.width) // 2
        by = max(6, bg.height - sprite.height - 10)
        bg.alpha_composite(sprite, (bx, by))
        alpha_bbox = sprite.getchannel("A").point(lambda p: 255 if p > 4 else 0).getbbox()
        if alpha_bbox:
            x0, y0, x1, y1 = alpha_bbox
            draw_bg = ImageDraw.Draw(bg)
            draw_bg.rectangle((bx + x0, by + y0, bx + x1 - 1, by + y1 - 1), outline=(35, 117, 102, 255), width=1)
        panel_x = (idx % 3) * cell_w + 12
        panel_y = (idx // 3) * cell_h + 10
        sheet.alpha_composite(bg, (panel_x, panel_y))
        draw.text((panel_x, cell_h * (idx // 3) + cell_h - 34), state, fill=(47, 42, 46, 255))
        if alpha_bbox:
            bbox_text = f"{alpha_bbox[2] - alpha_bbox[0]}x{alpha_bbox[3] - alpha_bbox[1]}"
            draw.text((panel_x, cell_h * (idx // 3) + cell_h - 18), bbox_text, fill=(117, 94, 76, 255))
    return sheet


def make_alpha_check(source: Image.Image, raw_sprite: Image.Image, base_sprite: Image.Image, contact: Image.Image) -> Image.Image:
    raw_bg = checkerboard(raw_sprite.size)
    raw_bg.alpha_composite(raw_sprite.convert("RGBA"))

    base_scale = 2
    base_big = base_sprite.resize(
        (base_sprite.width * base_scale, base_sprite.height * base_scale),
        Image.Resampling.NEAREST,
    )
    base_bg = checkerboard(base_big.size, block=12)
    base_bg.alpha_composite(base_big)

    contact_scale = 0.44
    contact_thumb = contact.resize(
        (round(contact.width * contact_scale), round(contact.height * contact_scale)),
        Image.Resampling.NEAREST,
    ).convert("RGBA")

    panels = [
        labeled_panel(source.convert("RGBA"), "source template"),
        labeled_panel(raw_bg, "masked template"),
        labeled_panel(base_bg, "fitted template x2"),
        labeled_panel(contact_thumb, "contact sheet"),
    ]
    gap = 20
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGBA", (width, height), (255, 249, 240, 255))
    x = 0
    for panel in panels:
        sheet.alpha_composite(panel, (x, 0))
        x += panel.width + gap
    return sheet


def render_gifs(frames: dict[str, list[Image.Image]]) -> None:
    preview_dir = QA_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for state, state_frames in frames.items():
        bg_frames = []
        for frame in state_frames:
            bg = Image.new("RGBA", frame.size, (255, 249, 240, 255))
            bg.alpha_composite(frame)
            bg_frames.append(bg.convert("P", palette=Image.Palette.ADAPTIVE))
        bg_frames[0].save(
            preview_dir / f"{state}.gif",
            save_all=True,
            append_images=bg_frames[1:],
            duration=PREVIEW_GIF_FRAME_MS,
            loop=0,
            disposal=2,
        )


def validate_atlas(atlas: Image.Image) -> dict[str, object]:
    result: dict[str, object] = {
        "ok": True,
        "width": atlas.width,
        "height": atlas.height,
        "expected_width": CELL_W * COLUMNS,
        "expected_height": CELL_H * len(ROWS),
        "rows": [],
        "warnings": [],
    }
    if atlas.size != (CELL_W * COLUMNS, CELL_H * len(ROWS)):
        result["ok"] = False
    alpha = atlas.getchannel("A")
    for row_idx, (state, frame_count) in enumerate(ROWS):
        row_info = {"state": state, "frame_count": frame_count, "cells": []}
        for col in range(COLUMNS):
            cell = alpha.crop((col * CELL_W, row_idx * CELL_H, (col + 1) * CELL_W, (row_idx + 1) * CELL_H))
            count = sum(cell.histogram()[1:])
            should_be_used = col < frame_count
            if should_be_used and count == 0:
                result["ok"] = False
            if not should_be_used and count != 0:
                result["ok"] = False
            row_info["cells"].append({"column": col, "nontransparent_pixels": count, "used": should_be_used})
        result["rows"].append(row_info)
    return result


def write_package() -> None:
    pet_json = {
        "id": "shian-helper",
        "displayName": "诗岸",
        "description": "静静坐在屏幕角落的安静陪伴式编码伙伴。",
        "spritesheetPath": "spritesheet.webp",
    }
    avatar_json = {
        **pet_json,
        "name": pet_json["displayName"],
        "spritesheet": pet_json["spritesheetPath"],
    }
    for target in (PACKAGE_DIR, CODEX_PACKAGE_DIR):
        (target / "pet.json").write_text(json.dumps(pet_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "avatar.json").write_text(json.dumps(avatar_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(FINAL_DIR / "spritesheet.webp", target / "spritesheet.webp")


def main() -> None:
    ensure_dirs()
    for stale in ("extracted-shian-front.png", "base-sprite.png"):
        stale_path = RUN_DIR / stale
        if stale_path.exists():
            stale_path.unlink()

    sample_source = load_rgb(POSE_SOURCE_FILES["idle"])
    sample_raw_sprite = apply_background_alpha(
        sample_source,
        grow=POSE_MASK_GROW.get("idle", 5),
        blur=POSE_MASK_BLUR.get("idle", 0.35),
    )
    sample_raw_sprite.save(RUN_DIR / "sample-template-idle.png")

    pose_sprites = load_pose_sprites()
    make_pose_sheet(pose_sprites).convert("RGB").save(QA_DIR / "pose-source-sheet.png")
    make_pose_sheet(pose_sprites).convert("RGB").save(QA_DIR / "action-template.png")
    make_edge_audit_sheet(pose_sprites).convert("RGB").save(QA_DIR / "edge-audit-sheet.png")

    frames = make_frames(pose_sprites)
    save_frames(frames)
    atlas = compose_atlas(frames)
    atlas.save(FINAL_DIR / "spritesheet.png")
    atlas.save(FINAL_DIR / "spritesheet.webp", lossless=True, method=6, exact=True)

    contact = make_contact_sheet(atlas)
    contact.convert("RGB").save(QA_DIR / "contact-sheet.png")
    contact.convert("RGB").save(QA_DIR / "action-summary.png")
    alpha_check = make_alpha_check(sample_source, sample_raw_sprite, pose_sprites["idle"], contact)
    alpha_check.convert("RGB").save(QA_DIR / "alpha-check.png")
    render_gifs(frames)

    validation = validate_atlas(atlas)
    (FINAL_DIR / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (QA_DIR / "review.json").write_text(
        json.dumps(
            {
                "ok": validation["ok"],
                "source": str(REF_DIR / POSE_SOURCE_FILES["idle"]),
                "method": "multi-frame atlas built from full action templates using background transparency plus live2d-style full-canvas mesh deformation",
                "note": "State motion is generated from whole-template deformation and global transforms, without subject cropping or cutout motion.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_package()
    (QA_DIR / "run-summary.json").write_text(
        json.dumps(
            {
                "ok": validation["ok"],
                "run_dir": str(RUN_DIR),
                "spritesheet": str(FINAL_DIR / "spritesheet.webp"),
                "validation": str(FINAL_DIR / "validation.json"),
                "contact_sheet": str(QA_DIR / "contact-sheet.png"),
                "package": str(PACKAGE_DIR),
                "codex_package": str(CODEX_PACKAGE_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": validation["ok"], "package": str(PACKAGE_DIR), "codex_package": str(CODEX_PACKAGE_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
