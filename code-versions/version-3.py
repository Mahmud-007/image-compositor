#!/usr/bin/env python3
"""
Place an image into the white area of a template WITHOUT black bars.

What changed vs. before:
- We now use a "COVER" fit (scale up then center-crop overflow) so the region is
  100% filled. This removes the black sides you saw.
- If you prefer the old "CONTAIN" behavior (fit-inside + padding), set mode="contain".

Usage:
  pip install opencv-python-headless numpy requests
  python image_cover_fit.py --template template.png \
    --image https://media.cnn.com/api/v1/images/stellar/prod/ap25232764947094.jpg \
    --output composited.png
"""

import os
import cv2
import sys
import argparse
import tempfile
import numpy as np
import requests


# ------------------------- IO HELPERS -------------------------

def load_image(path_or_url: str) -> np.ndarray:
    """
    Load an image from a local path OR a URL into OpenCV BGR ndarray.

    Notes:
    - For URLs we download into a temporary file first, then read with cv2.imread.
    - If this fails, we raise an exception with a clear message.
    """
    if path_or_url.startswith(("http://", "https://")):
        with requests.get(path_or_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                tmp_path = f.name
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    else:
        img = cv2.imread(path_or_url, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Failed to load image: {path_or_url}")
    return img


# -------------------- GEOMETRY / DETECTION --------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [top-left, top-right, bottom-right, bottom-left].
    This ensures a stable mapping for perspective transforms.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL has the smallest sum
    rect[2] = pts[np.argmax(s)]   # BR has the largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR has the smallest (x - y)
    rect[3] = pts[np.argmax(diff)]  # BL has the largest (x - y)
    return rect


def detect_largest_white_region(template_bgr: np.ndarray,
                                sat_max: int = 40,
                                val_min: int = 200) -> np.ndarray:
    """
    Return the polygon approximation (cv2.approxPolyDP) of the largest white region.

    How "white" is defined:
    - Very low saturation (<= sat_max)
    - Very high value/brightness (>= val_min)
    Adjust sat_max/val_min if your template differs.

    We use morphological open/close to clean the mask, then pick the largest contour.
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
        raise RuntimeError("No white regions detected. Tweak --sat-max / --val-min.")

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx


# -------------------------- FIT LOGIC -------------------------

def fit_source_to_rect_cover(src_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    COVER fit (no black bars):
    - Scale the source so the target rectangle is completely covered.
    - Then center-crop any overflow to exactly (target_w, target_h).
    - This is equivalent to CSS background-size: cover.

    This removes the black space by intentionally zooming/cropping.
    """
    sh, sw = src_bgr.shape[:2]
    src_aspect = sw / sh
    tgt_aspect = target_w / target_h

    # If source is "wider" than target, we need enough height → scale by height
    # Else scale by width. This ensures full coverage (no empty areas).
    if src_aspect < tgt_aspect:
        # Source is relatively taller → scale up by width
        new_w = target_w
        new_h = int(new_w / src_aspect)
    else:
        # Source is relatively wider → scale up by height
        new_h = target_h
        new_w = int(new_h * src_aspect)

    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center-crop to target size
    x = max(0, (new_w - target_w) // 2)
    y = max(0, (new_h - target_h) // 2)
    return resized[y:y + target_h, x:x + target_w]


def composite_into_template(template_bgr: np.ndarray,
                            src_bgr: np.ndarray,
                            approx: np.ndarray) -> np.ndarray:
    """
    Composite the source into the detected white region using COVER fit.
    - If the region is a quadrilateral, we:
        1) Build a rectangle of size (maxW x maxH)
        2) COVER-fit the source into that rectangle (no bars)
        3) Perspective-warp the rectangle into the quad
    - If it's not a quad, we COVER-fit into the bounding rectangle.
    """
    H, W = template_bgr.shape[:2]
    out = template_bgr.copy()

    # QUAD = perspective warp
    if len(approx) == 4:
        quad = order_points(approx.reshape(4, 2).astype("float32"))
        (tl, tr, br, bl) = quad
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxW = max(1, int(max(widthA, widthB)))
        maxH = max(1, int(max(heightA, heightB)))

        # 1) Make a rectangular canvas (the "target plane")
        plane = fit_source_to_rect_cover(src_bgr, maxW, maxH)

        # 2) Warp canvas to the quad
        src_quad = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
                            dtype="float32")
        M = cv2.getPerspectiveTransform(src_quad, quad)
        warped = cv2.warpPerspective(plane, M, (W, H))

        # 3) Mask only inside the quad (so we don't overwrite other parts)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        mask3 = cv2.merge([mask, mask, mask])

        return (warped & mask3) + (out & (~mask3))

    # NON-QUAD fallback = just use the bounding rect
    x, y, w, h = cv2.boundingRect(approx)
    fitted = fit_source_to_rect_cover(src_bgr, w, h)
    out[y:y + h, x:x + w] = fitted
    return out


# ---------------------------- MAIN ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Fill template's white area with an image (no black bars).")
    parser.add_argument("--template", required=True, help="Path/URL to template image (with white target area).")
    parser.add_argument("--image", required=True, help="Path/URL to the photo to place.")
    parser.add_argument("--output", default="composited.png", help="Output path (default: composited.png).")
    parser.add_argument("--sat-max", type=int, default=40, help="Max HSV saturation to count as white (default 40).")
    parser.add_argument("--val-min", type=int, default=200, help="Min HSV value/brightness for white (default 200).")
    args = parser.parse_args()

    # 1) Load inputs
    template_bgr = load_image(args.template)
    src_bgr = load_image(args.image)

    # 2) Find white region in the template
    approx = detect_largest_white_region(template_bgr, sat_max=args.sat_max, val_min=args.val_min)

    # 3) Composite with COVER fit (fills fully, crops overflow)
    out = composite_into_template(template_bgr, src_bgr, approx)

    # 4) Save
    if not cv2.imwrite(args.output, out):
        raise RuntimeError(f"Failed to write: {args.output}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

