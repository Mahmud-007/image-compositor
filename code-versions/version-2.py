#!/usr/bin/env python3
"""
image_compositor.py

Usage examples:
  # Local files
  python image_compositor.py --template template.png --image image-1.png

  # With a URL
  python image_compositor.py --template template.png \
      --image https://media.cnn.com/api/v1/images/stellar/prod/ap25232764947094.jpg

  # Custom output path
  python image_compositor.py --template template.png --image ... --output composited.png

  # If the white detection needs tweaking (S=0..40, V=200..255 by default):
  python image_compositor.py --sat-max 35 --val-min 210
"""
import os
import cv2
import sys
import argparse
import tempfile
import numpy as np
import requests


# -------- Helpers
def load_image(path_or_url: str) -> np.ndarray:
    """
    Load an image from a local path or URL into OpenCV BGR ndarray.
    Raises on failure.
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


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def detect_largest_white_region(template_bgr: np.ndarray,
                                sat_max: int = 40,
                                val_min: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mask, approx) where:
      - mask is a cleaned binary mask of white areas
      - approx is the polygon (cv2.approxPolyDP) of the largest white contour
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
        raise RuntimeError("No white regions detected. Try lowering --val-min or raising --sat-max.")

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return mask, approx


def fit_source_to_rect(src_bgr: np.ndarray, target_w: int, target_h: int,
                       letterbox: bool = True,
                       letterbox_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize src to fit into (target_w, target_h).
    If letterbox=True, pads to exactly target size while preserving aspect.
    If letterbox=False, fills the target by cropping the longer side (center-crop).
    """
    sh, sw = src_bgr.shape[:2]
    src_aspect = sw / sh
    tgt_aspect = target_w / target_h

    if letterbox:
        # Fit entirely inside, then pad
        if src_aspect > tgt_aspect:
            new_w = target_w
            new_h = int(new_w / src_aspect)
        else:
            new_h = target_h
            new_w = int(new_h * src_aspect)
        resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((target_h, target_w, 3), letterbox_color, dtype=np.uint8)
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas
    else:
        # Fill completely by cropping overflow
        if src_aspect > tgt_aspect:
            # Too wide → crop width
            new_h = target_h
            new_w = int(new_h * src_aspect)
        else:
            # Too tall → crop height
            new_w = target_w
            new_h = int(new_w / src_aspect)

        resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = (new_w - target_w) // 2
        y = (new_h - target_h) // 2
        return resized[y:y + target_h, x:x + target_w]


def composite_into_template(template_bgr: np.ndarray,
                            src_bgr: np.ndarray,
                            approx: np.ndarray,
                            letterbox: bool = True,
                            letterbox_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    If white region is a quad → perspective-warp into it.
    Else → fit inside bounding rect.
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

        fitted = fit_source_to_rect(src_bgr, maxW, maxH, letterbox=letterbox,
                                    letterbox_color=letterbox_color)

        src_quad = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
                            dtype="float32")
        M = cv2.getPerspectiveTransform(src_quad, quad)
        warped = cv2.warpPerspective(fitted, M, (W, H))

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        mask3 = cv2.merge([mask, mask, mask])

        out = (warped & mask3) + (out & (~mask3))
        return out

    # Fallback: bounding rectangle
    x, y, w, h = cv2.boundingRect(approx)
    fitted = fit_source_to_rect(src_bgr, w, h, letterbox=letterbox,
                                letterbox_color=letterbox_color)
    out[y:y + h, x:x + w] = fitted
    return out


# -------- Main
def main():
    ap = argparse.ArgumentParser(description="Place an image into the white area of a template.")
    ap.add_argument("--template", required=True,
                    help="Path or URL to the template image (with a white target area).")
    ap.add_argument("--image", required=True,
                    help="Path or URL to the photo to place into the template.")
    ap.add_argument("--output", default="composited.png",
                    help="Path to save the output image (default: composited.png).")
    ap.add_argument("--sat-max", type=int, default=40,
                    help="Max HSV saturation to consider 'white' (default: 40).")
    ap.add_argument("--val-min", type=int, default=200,
                    help="Min HSV value (brightness) to consider 'white' (default: 200).")
    ap.add_argument("--no-letterbox", action="store_true",
                    help="Fill the target by cropping instead of letterboxing.")
    ap.add_argument("--letterbox-color", default="0,0,0",
                    help="BGR color for letterbox padding, e.g., '255,255,255' for white.")
    args = ap.parse_args()

    # Parse letterbox color
    try:
        bgr = tuple(int(x) for x in args.letterbox_color.split(","))
        assert len(bgr) == 3 and all(0 <= c <= 255 for c in bgr)
    except Exception:
        print("Invalid --letterbox-color; expected 'B,G,R' with 0..255. Using 0,0,0.", file=sys.stderr)
        bgr = (0, 0, 0)

    # Load images
    template_bgr = load_image(args.template)
    src_bgr = load_image(args.image)

    # Detect white region and composite
    _, approx = detect_largest_white_region(template_bgr,
                                            sat_max=args.sat_max,
                                            val_min=args.val_min)
    out = composite_into_template(
        template_bgr,
        src_bgr,
        approx,
        letterbox=not args.no_letterbox,
        letterbox_color=bgr,
    )

    # Save
    ok = cv2.imwrite(args.output, out)
    if not ok:
        raise RuntimeError(f"Failed to write output to {args.output}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
