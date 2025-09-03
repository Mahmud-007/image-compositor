#!/usr/bin/env python3
"""
compose_with_template.py (version-5)

Fixes:
- Title/meta font sizes and font files are now honored (no silent bitmap fallback).
- You can pass --font-regular/--font-bold and --title-size/--meta-size and they will apply.

Usage (example):
python version-5.py \
  --template just-template.png \
  --image https://img.freepik.com/free-vector/collaborative-robotics-abstract-concept-illustration_335657-2115.jpg \
  --logo cnn-logo.png \
  --title "দনবাস ছাড়, ন্যাটোতে যাওয়া যাবে না, পশ্চিমা সেনাও চলবে না: ইউক্রেনকে পুতিনের শর্ত" \
  --date "22.08.2025" --source "Source: CNN" \
  --center "facebook.com/banaimedia" --right "banaimedia.com" \
  --font-regular "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" \
  --font-bold "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" \
  --title-size 70 --meta-size 22 \
  --output final.png
"""
import io
import os
import cv2
import sys
import argparse
import numpy as np
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------- robust TTF loader (forces real scalable fonts) ----------
COMMON_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
def load_ttf(font_path: str | None, size_px: int) -> ImageFont.FreeTypeFont:
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
        "No TTF found. Provide --font-regular/--font-bold with .ttf files."
    )

# ---------- IO helpers ----------
def load_cv_or_pil(path_or_url: str, as_pil: bool = False):
    def _read_bytes(p):
        if p.startswith(("http://", "https://")):
            r = requests.get(p, timeout=30); r.raise_for_status()
            return r.content
        with open(p, "rb") as f: return f.read()
    data = _read_bytes(path_or_url)
    if as_pil:
        return Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Failed to load: {path_or_url}")
    return img

def load_cv(path_or_url: str) -> np.ndarray:
    if path_or_url.startswith(("http://","https://")):
        r = requests.get(path_or_url, timeout=30); r.raise_for_status()
        arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_url, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Failed to load: {path_or_url}")
    return img

# ---------- geometry / detection ----------
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def detect_white_region(template_bgr: np.ndarray, sat_max=40, val_min=200):
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,val_min], np.uint8)
    upper_white = np.array([179,sat_max,255], np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: raise RuntimeError("White panel not found; tweak thresholds.")
    largest = max(cnts, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    x,y,w,h = cv2.boundingRect(largest)
    return approx, y, y+h

def fit_source_to_rect_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    sh, sw = src_bgr.shape[:2]
    src_as = sw / sh; tgt_as = tw / th
    if src_as < tgt_as: nw, nh = tw, int(tw / src_as)
    else:               nh, nw = th, int(th * src_as)
    resized = cv2.resize(src_bgr, (nw, nh), cv2.INTER_AREA)
    x, y = (nw - tw)//2, (nh - th)//2
    return resized[y:y+th, x:x+tw]

def place_photo(template_bgr, photo_bgr, approx):
    H,W = template_bgr.shape[:2]
    out = template_bgr.copy()
    if len(approx)==4:
        quad = order_points(approx.reshape(4,2).astype("float32"))
        (tl,tr,br,bl) = quad
        wA,wB = np.linalg.norm(br-bl), np.linalg.norm(tr-tl)
        hA,hB = np.linalg.norm(tr-br), np.linalg.norm(tl-bl)
        maxW,maxH = int(max(wA,wB)), int(max(hA,hB))
        plane = fit_source_to_rect_cover(photo_bgr, maxW, maxH)
        src_quad = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], "float32")
        M = cv2.getPerspectiveTransform(src_quad, quad)
        warped = cv2.warpPerspective(plane, M, (W,H))
        mask = np.zeros((H,W), np.uint8); cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        mask3 = cv2.merge([mask,mask,mask])
        return (warped & mask3) + (out & (~mask3))
    x,y,w,h = cv2.boundingRect(approx)
    out[y:y+h, x:x+w] = fit_source_to_rect_cover(photo_bgr, w, h)
    return out

# ---------- text helpers ----------
def draw_wrapped_text(draw, text, font, box_w, start_xy, line_spacing=0.95, fill=(255,255,255)):
    if not text: return 0
    words = text.split(); lines=[]; cur=""
    for w in words:
        test = (cur+" "+w).strip()
        if draw.textlength(test, font=font) <= box_w or not cur: cur=test
        else: lines.append(cur); cur=w
    if cur: lines.append(cur)
    x,y = start_xy
    ascent, descent = font.getmetrics(); line_h = ascent+descent
    for ln in lines:
        draw.text((x,y), ln, font=font, fill=fill)
        y += int(line_h * line_spacing)
    return y - start_xy[1]

# ---------- compose ----------
def compose(template_path, image_path, logo_path, title, date_text, source_text,
            center_text, right_text, output_path, *,
            font_regular="", font_bold="", title_size=140, meta_size=48):
    template_bgr = load_cv(template_path)
    approx, top_y, bottom_y = detect_white_region(template_bgr)
    photo_bgr = load_cv(image_path)
    composed_bgr = place_photo(template_bgr, photo_bgr, approx)

    # switch to PIL for crisp text/logo
    composed = Image.fromarray(cv2.cvtColor(composed_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(composed)
    W,H = composed.size
    top_band_h  = top_y
    bottom_band_h = H - bottom_y

    pad = int(0.035 * W); small_pad = int(pad*0.6)
    WHITE = (255,255,255)

    # --- load fonts using provided paths and sizes
    title_font = load_ttf(font_bold or font_regular, title_size)
    meta_font  = load_ttf(font_regular or font_bold, meta_size)

    # logo in top-left
    if logo_path:
        try:
            logo = load_cv_or_pil(logo_path, as_pil=True)
            target_h = max(1, int(top_band_h * 0.7))
            scale = target_h / max(1, logo.height)
            logo = logo.resize((int(logo.width*scale), target_h), Image.LANCZOS)
            composed.paste(logo, (pad, (top_band_h - target_h)//2), mask=logo if logo.mode=="RGBA" else None)
        except Exception as e:
            print(f"Logo skipped: {e}", file=sys.stderr)

    # title in bottom band
    title_left = pad
    title_top  = bottom_y + small_pad
    title_width = W - 2*pad
    draw_wrapped_text(draw, title, title_font, title_width, (title_left, title_top),
                      line_spacing=0.95, fill=WHITE)

    # footer row
    footer_y = H - small_pad - meta_font.getbbox("Ag")[3]
    left_text = (date_text + "  " + source_text).strip()
    draw.text((pad, footer_y), left_text, font=meta_font, fill=WHITE)
    center_w = draw.textlength(center_text, font=meta_font)
    draw.text(((W-center_w)/2, footer_y), center_text, font=meta_font, fill=WHITE)
    right_w = draw.textlength(right_text, font=meta_font)
    draw.text((W - pad - right_w, footer_y), right_text, font=meta_font, fill=WHITE)

    composed.save(output_path)
    print(f"Saved: {output_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Fill template and add logo/title/footer texts.")
    ap.add_argument("--template", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--logo", default="")
    ap.add_argument("--title", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--center", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--output", default="final.png")
    ap.add_argument("--font-regular", default="", help="Path to Regular .ttf")
    ap.add_argument("--font-bold",    default="", help="Path to Bold .ttf")
    ap.add_argument("--title-size",   type=int, default=140, help="Title font size (px)")
    ap.add_argument("--meta-size",    type=int, default=48,  help="Footer font size (px)")
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
        font_regular=args.font_regular,
        font_bold=args.font_bold,
        title_size=args.title_size,
        meta_size=args.meta_size,
    )

if __name__ == "__main__":
    main()
