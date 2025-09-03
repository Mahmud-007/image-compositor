#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Domka Bangla compositor

Usage:
python domka-bangla-2.py \
  --template "Domka-Bangla-template-1.jpg" \
  --image "https://media.prothomalo.com/prothomalo-bangla%2F2025-08-28%2F9ok3cfdf%2Fprothomalo-bangla-Latif.avif" \
  --output domka-bangla-2-output.png
"""

import io, os, cv2, argparse, requests
import numpy as np
from PIL import Image

try:
    import pillow_avif  # allows .avif support if installed
except Exception:
    pass


def load_cv(path_or_url: str) -> np.ndarray:
    """Load an image from URL or path into OpenCV BGR format."""
    data = None
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

    # fallback to PIL for AVIF/odd formats
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fit_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Scale src to fully cover (tw x th), then crop center."""
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


def detect_black_box(template_bgr: np.ndarray) -> tuple[int,int,int,int]:
    """Find the big black rectangle (photo placeholder)."""
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    # shrink by 2px to avoid border overlap
    return x+2, y+2, w-4, h-4


def compose(template: str, image: str, output: str):
    """Place the image inside the black box of the template and save."""
    template_bgr = load_cv(template)
    x, y, w, h = detect_black_box(template_bgr)
    photo_bgr = load_cv(image)
    fitted = fit_cover(photo_bgr, w, h)
    template_bgr[y:y+h, x:x+w] = fitted
    cv2.imwrite(output, template_bgr)
    print(f"✅ Saved: {output}")


def main():
    ap = argparse.ArgumentParser(description="Simple Domka Bangla compositor")
    ap.add_argument("--template", required=True, help="Template image (path or URL)")
    ap.add_argument("--image", required=True, help="Actual photo (path or URL)")
    ap.add_argument("--output", required=True, help="Output file name")
    args = ap.parse_args()
    compose(args.template, args.image, args.output)


if __name__ == "__main__":
    main()
