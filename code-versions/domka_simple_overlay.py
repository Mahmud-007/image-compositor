#!/usr/bin/env python3
# Usage:
# python domka_simple_overlay.py \
#   --template "Domka-Bangla-template-1.jpg" \
#   --image "https://media.prothomalo.com/prothomalo-bangla%2F2025-08-28%2F9ok3cfdf%2Fprothomalo-bangla-Latif.avif" \
#   --output domka_simple_overlay.png

import io, os, cv2, argparse, requests
import numpy as np
from PIL import Image

# Optional: add AVIF support if available
try:
    import pillow_avif  # noqa: F401
except Exception:
    pass


def load_cv(path_or_url: str) -> np.ndarray:
    """Load an image (URL or path) into OpenCV BGR. Falls back to PIL for AVIF/odd formats."""
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

    # Fallback via PIL → RGB → BGR
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fit_cover(src_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Scale to fully cover (tw x th), crop center — like CSS background-size: cover."""
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


def detect_black_box(template_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the big black rectangle (photo placeholder) on the template.
    We threshold 'very dark' in HSV, then take the largest contour.
    """
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([179, 80, 60], np.uint8)  # allow dark gray too
    mask = cv2.inRange(hsv, lower, upper)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No black rectangle found in template.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))

    # shrink a couple pixels so the image never covers frame strokes
    return x + 2, y + 2, max(1, w - 4), max(1, h - 4)


def compose(template: str, image: str, output: str):
    """
    1) Place the photo inside the detected black box (cover fit).
    2) Overlay *non-black* pixels from the original template BACK ON TOP.
       → This restores the on-top logo and the category tag that overlaps the photo edge.
    """
    # Load original template twice (we need a clean copy for overlay)
    template_bgr = load_cv(template)
    overlay_src  = load_cv(template)

    # Detect photo box and fill with the fitted image
    x, y, w, h = detect_black_box(template_bgr)
    photo_bgr   = load_cv(image)
    fitted      = fit_cover(photo_bgr, w, h)
    base = template_bgr.copy()
    base[y:y+h, x:x+w] = fitted  # photo goes into the box

    # Build overlay mask of "non-black" pixels from the template
    hsv_overlay = cv2.cvtColor(overlay_src, cv2.COLOR_BGR2HSV)
    # black if V is very low; everything else is "non-black" (logo, category pill, gradients, strokes)
    black_mask  = cv2.inRange(hsv_overlay, np.array([0, 0, 0], np.uint8),
                                            np.array([179, 80, 60], np.uint8))
    nonblack_mask = cv2.bitwise_not(black_mask)  # what we want to paste on top

    # 3-channel mask
    mask3 = cv2.merge([nonblack_mask, nonblack_mask, nonblack_mask])

    # Composite: keep 'base' where mask is 0; use 'overlay_src' where mask is 255
    # This draws logo/category/frame ABOVE the photo.
    final = (base & (~mask3)) + (overlay_src & mask3)

    cv2.imwrite(output, final)
    print(f"✅ Saved: {output}")


def main():
    ap = argparse.ArgumentParser(description="Minimal Domka Bangla compositor with top overlays")
    ap.add_argument("--template", required=True, help="Template image (path or URL)")
    ap.add_argument("--image", required=True, help="Actual photo (path or URL)")
    ap.add_argument("--output", required=True, help="Output file name")
    args = ap.parse_args()
    compose(args.template, args.image, args.output)


if __name__ == "__main__":
    main()
