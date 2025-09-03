#!/usr/bin/env python3
"""
compose_with_template.py

- Fills the white panel of just-template.png with a given image (cover-fit: no black bars).
- Adds overlays:
    • top-left logo image inside the top yellow band
    • big title in the bottom yellow band (auto-wrapped, left-aligned)
    • footer row on the very bottom: left (date+source), center (text), right (text)

USAGE EXAMPLE
-------------
pip install opencv-python-headless pillow numpy requests

python version-4.py --template just-template.png --image https://media.cnn.com/api/v1/images/stellar/prod/ap25232764947094.jpg --logo cnn-logo.png --title "Test Title:China snaps up Russian oil as Indian demand drops following Trump tariffs" --date "22.08.2025" --source "Source: CNN" --center "facebook.com/banaimedia" --right "banaimedia.com" --output final.png

python version-4.py --template just-template.png --image https://media.cnn.com/api/v1/images/stellar/prod/ap25232764947094.jpg --logo cnn-logo.png --title "Test Title:China snaps up Russian oil as Indian demand drops following Trump tariffs" --date "22.08.2025" --source "Source: CNN" --center "facebook.com/banaimedia" --right "banaimedia.com" --output final-1.png --title-size 200 --font-regular "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" --meta-size 64  --font-bold "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" 
"""

import os
import io
import cv2
import sys
import math
import argparse
import tempfile
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
from pathlib import Path

COMMON_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

def load_ttf(font_path: str | None, size_px: int) -> ImageFont.FreeTypeFont:
    """
    Load a TrueType font with the given size.
    If font_path is None, try several common system locations.
    If none exist, raise (so we don't silently use PIL's bitmap default).
    """
    candidates: list[str] = []
    if font_path:
        candidates.append(font_path)
    candidates += COMMON_FONT_PATHS

    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size_px)
        except Exception:
            pass

    raise RuntimeError(
        "No TTF font found. Provide --font-regular/--font-bold with paths to .ttf files."
    )
# ------------------------- Helpers -------------------------

def load_cv_or_pil(path_or_url: str, as_pil: bool = False):
    """
    Load an image from local path OR URL.
    - If as_pil=True -> return a PIL.Image in RGB
    - else -> return a cv2 image (BGR)
    """
    def _read_bytes(p):
        if p.startswith(("http://", "https://")):
            r = requests.get(p, timeout=30)
            r.raise_for_status()
            return r.content
        else:
            with open(p, "rb") as f:
                return f.read()

    data = _read_bytes(path_or_url)
    if as_pil:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    else:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path_or_url}")
        return img


def load_cv(path_or_url: str) -> np.ndarray:
    # OpenCV BGR
    if path_or_url.startswith(("http://", "https://")):
        with requests.get(path_or_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            arr = np.asarray(bytearray(r.content), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_url, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path_or_url}")
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL for stable perspective mapping."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_white_region(template_bgr: np.ndarray, sat_max=40, val_min=200):
    """
    Return (approx, top_y, bottom_y) where:
      - approx is polygon (cv2.approxPolyDP) for the largest white region
      - top_y is the top boundary (y) of the white region
      - bottom_y is the bottom boundary (y) of the white region
    We use it to infer the yellow bands above and below.
    """
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, val_min], dtype=np.uint8)
    upper_white = np.array([179, sat_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No white region detected—tweak thresholds.")

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    x, y, w, h = cv2.boundingRect(largest)
    top_y = y
    bottom_y = y + h
    return approx, top_y, bottom_y


def fit_source_to_rect_cover(src_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    COVER fit (no black bars): scale then center-crop to (target_w, target_h).
    """
    sh, sw = src_bgr.shape[:2]
    src_aspect = sw / sh
    tgt_aspect = target_w / target_h

    if src_aspect < tgt_aspect:
        new_w = target_w
        new_h = int(new_w / src_aspect)
    else:
        new_h = target_h
        new_w = int(new_h * src_aspect)

    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (new_w - target_w) // 2
    y = (new_h - target_h) // 2
    return resized[y:y + target_h, x:x + target_w]


def place_photo(template_bgr: np.ndarray, photo_bgr: np.ndarray, approx: np.ndarray) -> np.ndarray:
    """
    Put the photo into the white region. If region is a quad, perspective warp;
    else fill the bounding rectangle.
    """
    H, W = template_bgr.shape[:2]
    out = template_bgr.copy()

    if len(approx) == 4:
        quad = order_points(approx.reshape(4, 2).astype("float32"))
        (tl, tr, br, bl) = quad
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxW = max(1, int(max(widthA, widthB)))
        maxH = max(1, int(max(heightA, heightB)))

        plane = fit_source_to_rect_cover(photo_bgr, maxW, maxH)
        src_quad = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_quad, quad)
        warped = cv2.warpPerspective(plane, M, (W, H))

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        mask3 = cv2.merge([mask, mask, mask])

        return (warped & mask3) + (out & (~mask3))

    # Fallback for non-quad shapes
    x, y, w, h = cv2.boundingRect(approx)
    plane = fit_source_to_rect_cover(photo_bgr, w, h)
    out[y:y + h, x:x + w] = plane
    return out


def get_font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Load a TTF font. If bold=True, load a bold font file.
    """
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size_px)
        return ImageFont.truetype("DejaVuSans.ttf", size_px)
    except Exception:
        return ImageFont.load_default()


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                      box_w: int, start_xy: tuple[int, int], line_spacing: float = 1.05,
                      fill=(255, 255, 255)) -> int:
    """
    Draw left-aligned wrapped text into a width-limited box.
    Returns the total height in pixels used.
    """
    if not text:
        return 0

    # Greedy wrap by measuring words
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= box_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    x, y = start_xy
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    for i, line in enumerate(lines):
        draw.text((x, y), line, font=font, fill=fill)
        y += int(line_h * line_spacing)

    return y - start_xy[1]


# ------------------------- Main Compose -------------------------

def compose(template_path, image_path, logo_path, title, date_text, source_text,
            center_text, right_text, output_path, *,
            font_regular="", font_bold="", title_size=140, meta_size=48):
    # 1) Load template + detect white region and bands
    template_bgr = load_cv(template_path)
    approx, top_y, bottom_y = detect_white_region(template_bgr)

    # 2) Load the photo and place it into the white area
    photo_bgr = load_cv(image_path)
    composed_bgr = place_photo(template_bgr, photo_bgr, approx)

    # 3) Switch to PIL for high-quality text/logo drawing
    composed = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(composed)
    W, H = composed.size

    # Yellow bands (computed from white region bounds)
    top_band_height = top_y
    bottom_band_height = H - bottom_y

    # --- Layout constants (tweak as needed)
    pad = int(0.035 * W)             # general padding relative to width
    small_pad = int(pad * 0.6)

    # Title font size relative to bottom band height
    title_font = get_font(140, bold=True)   # huge fixed font
    meta_font  = get_font(48) 

    # Colors (white text works well on the orange band)
    WHITE = (255, 255, 255)

    # 4) Logo: placed in the top-left inside the top band
    if logo_path:
        try:
            logo = load_cv_or_pil(logo_path, as_pil=True)
            # Fit logo height to ~70% of top band, keep aspect; clamp width
            target_h = max(1, int(top_band_height * 0.7))
            scale = target_h / max(1, logo.height)
            target_w = int(logo.width * scale)
            logo = logo.resize((target_w, target_h), Image.LANCZOS)
            # Keep some horizontal padding
            composed.paste(logo, (pad, (top_band_height - target_h) // 2), mask=logo if logo.mode == "RGBA" else None)
        except Exception as e:
            print(f"Logo load/draw skipped: {e}", file=sys.stderr)

    # 5) Title: big, left-aligned on the bottom band; wrap to available width
    title_left = pad
    title_top = bottom_y + small_pad
    title_width = W - 2 * pad
    draw_wrapped_text(draw, title, title_font, title_width, (title_left, title_top), line_spacing=1.05, fill=WHITE)

    # 6) Footer row (small text) near the very bottom:
    #    left: "date • source", center: center_text, right: right_text
    footer_y = H - small_pad - meta_font.getbbox("Ag")[3]  # baseline from the bottom
    left_text = (date_text + "  " + source_text).strip()

    # Left
    draw.text((pad, footer_y), left_text, font=meta_font, fill=WHITE)
    # Center (we center by measuring width)
    center_w = draw.textlength(center_text, font=meta_font)
    draw.text(((W - center_w) / 2, footer_y), center_text, font=meta_font, fill=WHITE)
    # Right (right-aligned by measuring width)
    right_w = draw.textlength(right_text, font=meta_font)
    draw.text((W - pad - right_w, footer_y), right_text, font=meta_font, fill=WHITE)

    # 7) Save
    composed.save(output_path)
    print(f"Saved: {output_path}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Fill just-template.png and add logo/title/footer texts.")
    ap.add_argument("--template", required=True, help="Path/URL to just-template.png")
    ap.add_argument("--image", required=True, help="Path/URL to the photo for the white area")
    ap.add_argument("--logo", default="", help="Path/URL to a logo image (optional)")
    ap.add_argument("--title", required=True, help="Big title text")
    ap.add_argument("--date", required=True, help="Date text (left footer)")
    ap.add_argument("--source", required=True, help="Source text (left footer)")
    ap.add_argument("--center", required=True, help="Center footer text")
    ap.add_argument("--right", required=True, help="Right footer text")
    ap.add_argument("--output", default="final.png", help="Output file path")
    ap.add_argument("--font-regular", default="", help="Path to a regular TTF (e.g., DejaVuSans.ttf)")
    ap.add_argument("--font-bold",    default="", help="Path to a bold TTF (e.g., DejaVuSans-Bold.ttf)")
    ap.add_argument("--title-size",   type=int, default=140, help="Title font size in px (default 140)")
    ap.add_argument("--meta-size",    type=int, default=48,  help="Footer font size in px (default 48)")

    args = ap.parse_args()

    compose(
        template_path=args.template,
        image_path=args.image,
        logo_path=args.logo,
        title=args.title,
        date_text=args.date,
        source_text=args.source,
        center_text=args.center,
        right_text=args.right,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
