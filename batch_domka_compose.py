#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import csv
import cv2
import requests
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import pillow_avif  # noqa: F401  # optional
except Exception:
    pass

# ---------------- CONFIG ----------------
template_dir = Path("./templates")   # your template folder
csv_path     = Path("./prothomalo_31-08-2025.csv")  # your csv file
out_dir      = Path("./output")      # output folder
naming_mode  = "index"               # "index" or "title"
# ----------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

def load_cv(path_or_url: str) -> np.ndarray:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        data = r.content
    else:
        with open(path_or_url, "rb") as f:
            data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def detect_black_box(template_bgr: np.ndarray):
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x + 2, y + 2, max(1, w - 4), max(1, h - 4)

def fit_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    sh, sw = src_bgr.shape[:2]
    src_as = sw / sh
    tgt_as = tw / th
    if src_as < tgt_as:
        new_w = tw
        new_h = int(new_w / src_as)
    else:
        new_h = th
        new_w = int(new_h * src_as)
    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (new_w - tw) // 2
    y = (new_h - th) // 2
    return resized[y:y+th, x:x+tw]

def overlay_nonblack_on_top(base_bgr: np.ndarray, template_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0], np.uint8),
                                   np.array([179, 80, 60], np.uint8))
    nonblack_mask = cv2.bitwise_not(black_mask)
    mask3 = cv2.merge([nonblack_mask, nonblack_mask, nonblack_mask])
    return (base_bgr & (~mask3)) + (template_bgr & mask3)

def safe_slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return s[:max_len] or "untitled"

def list_templates_sorted(template_dir: Path):
    files = [p for p in template_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No template images in {template_dir}")
    files.sort(key=lambda p: int(re.search(r"\d+", p.stem).group(0)) if re.search(r"\d+", p.stem) else 9999)
    return files

def read_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def process_one(template_path: Path, article_image: str, out_path: Path):
    template_bgr = load_cv(str(template_path))
    overlay_src  = template_bgr.copy()
    x, y, w, h = detect_black_box(template_bgr)
    photo_bgr  = load_cv(article_image)
    fitted     = fit_cover(photo_bgr, w, h)
    base = template_bgr.copy()
    base[y:y+h, x:x+w] = fitted
    final_bgr = overlay_nonblack_on_top(base, overlay_src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final_bgr)

def main():
    templates = list_templates_sorted(template_dir)
    rows = read_csv_rows(csv_path)
    if len(templates) != len(rows):
        raise ValueError(f"Template count ({len(templates)}) != CSV rows ({len(rows)})")

    for i, (tpl, row) in enumerate(zip(templates, rows), start=1):
        img_url = (row.get("article_image") or "").strip()
        if not img_url:
            continue
        if naming_mode == "title":
            title = (row.get("title") or "").strip()
            base_name = safe_slug(title) if title else str(i)
        else:
            base_name = str(i)
        out_path = out_dir / f"{base_name}_final.png"
        print(f"[{i}/{len(rows)}] {tpl.name} <-- {img_url}")
        process_one(tpl, img_url, out_path)

    print(f"✅ Done. Results saved in {out_dir.resolve()}")

if __name__ == "__main__":
    main()
