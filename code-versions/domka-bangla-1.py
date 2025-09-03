#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domka Bangla template compositor

What it does
------------
1) Detect the big black rectangle (photo placeholder) in the template.
2) Download/load your image (path or URL), cover-fit (zoom/crop) into that shape.
3) (Optional) Add Bengali texts:
   - Category pill just touching the bottom edge of the photo
   - Big headline below
   - Reporter line
   - Date + Subline

Why "cover-fit"?
----------------
It scales the image so the target area is fully covered (no black bars), then center-crops overflow.

Dependencies
------------
pip install opencv-python-headless pillow numpy requests pillow-avif-plugin

If you don’t need AVIF, you can skip pillow-avif-plugin.
"""

import io
import os
import cv2
import sys
import math
import argparse
import numpy as np
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Try to load AVIF plugin if installed (safe no-op if missing)
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


# --------------------------- Robust font loader ---------------------------

COMMON_FONT_PATHS = [
    # Noto Sans Bengali (recommended for Bengali)
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf",
    # DejaVu / Arial fallbacks
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/Nirmala.ttf",          # Windows Bengali-ish fallback
    "C:/Windows/Fonts/nirmalas.ttf",         # Nirmala UI Semilight
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

def load_ttf(font_path: str | None, size_px: int) -> ImageFont.FreeTypeFont:
    """
    Load a TrueType font. If path not given, try common system paths.
    Raise if none found — we don't want PIL bitmap fallback (fixed size).
    """
    candidates = []
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
        "No scalable TTF found. Provide --font-regular/--font-bold pointing to .ttf files."
    )


# ----------------------------- Image IO helpers -----------------------------

def load_cv(path_or_url: str) -> np.ndarray:
    """
    Load image to OpenCV (BGR). Supports URL or path via PIL->numpy conversion
    so we can handle AVIF too when pillow-avif-plugin is present.
    """
    data = None
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        data = r.content
    else:
        with open(path_or_url, "rb") as f:
            data = f.read()

    # Try OpenCV directly first
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # Fallback: PIL (helps with AVIF/odd formats), then convert to BGR
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)  # H x W x 3 (RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# ------------------------------- Core geometry -------------------------------

def fit_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """
    COVER fit: scale to fully cover target (tw x th), center-crop overflow.
    """
    sh, sw = src_bgr.shape[:2]
    src_as = sw / sh
    tgt_as = tw / th

    if src_as < tgt_as:
        # relatively taller → scale by width
        new_w = tw
        new_h = int(new_w / src_as)
    else:
        # relatively wider → scale by height
        new_h = th
        new_w = int(new_h * src_as)

    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (new_w - tw) // 2
    y = (new_h - th) // 2
    return resized[y:y+th, x:x+tw]


def detect_black_photo_box(template_bgr: np.ndarray) -> tuple[int,int,int,int]:
    """
    Detect the large black rectangle (the photo placeholder).
    Returns bounding box (x, y, w, h).

    Heuristic:
    - Convert to HSV
    - Threshold for very low value (V) and low saturation (near black)
    - Morph clean, pick largest rectangle-ish contour in the center zone
    """
    H, W = template_bgr.shape[:2]
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)

    # Very dark (tweak if your template differs)
    lower = np.array([0, 0, 0], dtype=np.uint8)
    upper = np.array([179, 80, 60], dtype=np.uint8)  # allow a bit of dark grey
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Could not find the black photo placeholder. Adjust thresholds.")

    # Prefer largest by area but also roughly in the middle third of the page
    best = None
    best_score = -1
    cx_target, cy_target = W/2, H*0.40  # expected region center (slightly upper half)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        cx, cy = x+w/2, y+h/2
        # distance penalty from expected center
        dist = math.hypot(cx - cx_target, cy - cy_target)
        score = area - 3.0*dist  # weight area high, penalize off-center
        if score > best_score:
            best, best_score = (x,y,w,h), score

    x,y,w,h = best
    # Slightly deflate to avoid 1px borders
    pad = 2
    x = max(0, x+pad); y = max(0, y+pad)
    w = max(1, w-2*pad); h = max(1, h-2*pad)
    return x, y, w, h


# ------------------------------- Text helpers --------------------------------

def round_rect(draw: ImageDraw.ImageDraw, xy, radius, fill):
    """Draw a filled rounded rectangle."""
    x1,y1,x2,y2 = xy
    draw.rounded_rectangle([x1,y1,x2,y2], radius=radius, fill=fill)

def draw_wrapped_text(draw, text, font, box_w, start_xy, line_spacing=1.05, fill=(0,0,0), align="left"):
    """
    Simple greedy word-wrap using Pillow's textlength measurement.
    Returns bottom y after drawing.
    """
    if not text:
        return start_xy[1]

    words = text.split()
    lines, cur = [], ""
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
    for line in lines:
        if align == "center":
            w = draw.textlength(line, font=font)
            draw.text((x + (box_w - w)/2, y), line, font=font, fill=fill)
        else:
            draw.text((x, y), line, font=font, fill=fill)
        y += int(line_h * line_spacing)
    return y


# --------------------------------- Compose -----------------------------------

def compose(
    template: str,
    image: str,
    output: str,
    category: str = "",
    title: str = "",
    reporter: str = "",
    date_text: str = "",
    subline: str = "",
    font_regular: str = "",
    font_bold: str = "",
    title_size: int = 84,
    category_size: int = 36,
    meta_size: int = 36,
    brand_red: tuple[int,int,int] = (198, 52, 42),  # RGB approximate of the brand red
):
    """
    Main pipeline:
    - place photo into detected black box
    - add optional texts (category pill, headline, reporter, date, subline)
    """

    # 1) Load template and detect photo box
    template_bgr = load_cv(template)
    x, y, w, h = detect_black_photo_box(template_bgr)

    # 2) Load your image and cover-fit into the box
    src_bgr = load_cv(image)
    fitted = fit_cover(src_bgr, w, h)
    template_bgr[y:y+h, x:x+w] = fitted

    # 3) Switch to PIL for crisp text rendering
    canvas = Image.fromarray(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    W, H = canvas.size

    # 4) Load fonts (force scalable TTF)
    font_title = load_ttf(font_bold or font_regular, title_size)
    font_meta  = load_ttf(font_regular or font_bold, meta_size)
    font_cat   = load_ttf(font_bold or font_regular, category_size)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED   = brand_red  # brand-ish red

    # 5) Category pill (centered horizontally, overlapping the bottom of the photo area)
    if category:
        pill_text = category.strip()
        text_w = draw.textlength(pill_text, font=font_cat)
        pad_x = 32
        pad_y = 10
        pill_w = int(text_w + pad_x * 2)
        pill_h = int(font_cat.getbbox("Ag")[3] + pad_y * 2)
        pill_x = int(x + (w - pill_w) / 2)
        pill_y = int(y + h - pill_h / 2)  # half-overlap with photo bottom

        round_rect(draw, (pill_x, pill_y, pill_x+pill_w, pill_y+pill_h), radius=16, fill=RED)
        draw.text((pill_x + pad_x, pill_y + pad_y - 4), pill_text, font=font_cat, fill=WHITE)

    # 6) Headline (big red text), left/right margins relative to width
    if title:
        margin = int(W * 0.07)
        title_top = y + h + int(H * 0.02)  # a bit below the photo/pill
        box_w = W - 2 * margin
        draw_wrapped_text(draw, title, font_title, box_w, (margin, title_top),
                          line_spacing=1.06, fill=RED, align="left")

    # 7) Reporter, date, subline area (centered) — placed below title area
    # Rough vertical placements based on the sample composition
    info_center_x = W // 2
    info_y = y + h + int(H * 0.21)  # adjust as needed for your exact template

    if reporter:
        rep_w = draw.textlength(reporter, font=font_meta)
        draw.text((info_center_x - rep_w/2, info_y), reporter, font=font_meta, fill=RED)
        info_y += int(font_meta.getbbox("Ag")[3] * 1.4)

    if date_text:
        date_w = draw.textlength(date_text, font=font_meta)
        draw.text((info_center_x - date_w/2, info_y), date_text, font=font_meta, fill=BLACK)
        info_y += int(font_meta.getbbox("Ag")[3] * 1.1)

    if subline:
        sub_w = draw.textlength(subline, font=font_meta)
        draw.text((info_center_x - sub_w/2, info_y), subline, font=font_meta, fill=BLACK)

    # 8) Save
    canvas.save(output)
    print(f"Saved: {output}")


# ----------------------------------- CLI -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compose a Domka Bangla style post from template + image.")
    ap.add_argument("--template", required=True, help="Path to the Domka template image (e.g., Domka-Bangla-template-1.jpg)")
    ap.add_argument("--image", required=True, help="Path or URL to the photo (JPG/PNG/WEBP/AVIF)")
    ap.add_argument("--output", default="final.png", help="Where to save the final image")
    # Optional texts
    ap.add_argument("--category", default="", help="Category label (e.g., আন্তর্জাতিক)")
    ap.add_argument("--title", default="", help="Big headline (Bengali supported)")
    ap.add_argument("--reporter", default="", help="Reporter name (centered)")
    ap.add_argument("--date", dest="date_text", default="", help="Date text")
    ap.add_argument("--subline", default="", help="Subline under date")
    # Fonts & sizes
    ap.add_argument("--font-regular", default="", help="Path to a Bengali-capable TTF (e.g., NotoSansBengali-Regular.ttf)")
    ap.add_argument("--font-bold", default="", help="Path to a Bold TTF (e.g., NotoSansBengali-Bold.ttf)")
    ap.add_argument("--title-size", type=int, default=84, help="Title font size in px")
    ap.add_argument("--category-size", type=int, default=36, help="Category pill font size in px")
    ap.add_argument("--meta-size", type=int, default=36, help="Meta font size (reporter/date/subline) in px")
    args = ap.parse_args()

    compose(
        template=args.template,
        image=args.image,
        output=args.output,
        category=args.category,
        title=args.title,
        reporter=args.reporter,
        date_text=args.date_text,
        subline=args.subline,
        font_regular=args.font_regular,
        font_bold=args.font_bold,
        title_size=args.title_size,
        category_size=args.category_size,
        meta_size=args.meta_size,
    )


if __name__ == "__main__":
    main()

# python domka-bangla-1.py \
#   --template Domka-Bangla-template-1.jpg \
#   --image "https://media.prothomalo.com/prothomalo-bangla%2F2025-08-28%2F9ok3cfdf%2Fprothomalo-bangla-Latif.avif" \

#   --output domka-bangla-1-output.png

#     --category "আন্তর্জাতিক" \
#   --title "পৃথিবীর কোন এক প্রান্তে কোন এক সময় একটি ঘটনা ঘটেছে..." \
#   --reporter "বিস্তারিত মন্তব্য" \
#   --date "০১.০৯.২০২৫" \
#   --subline "উপস্থাপনায় ধরণেশ গুপ্ত" \
#   --font-regular "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf" \
#   --font-bold "/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf" \
#   --title-size 96 \
#   --category-size 40 \
#   --meta-size 38 \