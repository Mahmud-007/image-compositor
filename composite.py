# Install and import required libraries
import cv2
import numpy as np
from PIL import Image

# Paths provided by the user/developer
template_path = "./template.png"
image_path = "./judge-frank.jpg"
output_path = "./composited.png"

# --- Load images
template_bgr = cv2.imread(template_path)
image_bgr = cv2.imread(image_path)

if template_bgr is None or image_bgr is None:
    raise FileNotFoundError("One of the image paths is incorrect or the images could not be loaded.")

h_t, w_t = template_bgr.shape[:2]

# --- Find the largest "white" area in the template
# Convert to HSV for robust brightness/whiteness thresholding
hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)

# White tends to have very low saturation and very high value
# Tune thresholds if needed
lower_white = np.array([0, 0, 200], dtype=np.uint8)
upper_white = np.array([179, 40, 255], dtype=np.uint8)
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Morphology to clean up
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and pick the largest by area
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise RuntimeError("No white regions detected in the template. Adjust thresholds.")

largest = max(contours, key=cv2.contourArea)
area = cv2.contourArea(largest)

# Approximate the polygon of the white region
epsilon = 0.01 * cv2.arcLength(largest, True)
approx = cv2.approxPolyDP(largest, epsilon, True)

# Create a 4-point target (ordered) if the region looks like a quadrilateral; otherwise use bounding rect
def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

composited = template_bgr.copy()

if len(approx) == 4:
    # Perspective fit
    target_quad = order_points(approx.reshape(4, 2).astype("float32"))

    # Compute target width/height from the quad
    (tl, tr, br, bl) = target_quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Prepare the source image with the same aspect ratio as target
    src_h, src_w = image_bgr.shape[:2]
    # Resize source to target rectangle while preserving aspect ratio by letterboxing
    target_aspect = maxWidth / maxHeight if maxHeight > 0 else src_w / src_h
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # Fit by width, pad vertically
        new_w = maxWidth
        new_h = int(new_w / src_aspect)
    else:
        # Fit by height, pad horizontally
        new_h = maxHeight
        new_w = int(new_h * src_aspect)

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create letterboxed canvas of target size with black (or white) background
    canvas = np.zeros((maxHeight, maxWidth, 3), dtype=np.uint8)
    # Place resized centered
    y_off = (maxHeight - new_h) // 2
    x_off = (maxWidth - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # Now warp this canvas into the target quad
    src_quad = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_quad, target_quad)
    warped = cv2.warpPerspective(canvas, M, (w_t, h_t))

    # Create a mask for the quad and composite
    mask = np.zeros((h_t, w_t), dtype=np.uint8)
    cv2.fillConvexPoly(mask, target_quad.astype(np.int32), 255)
    mask3 = cv2.merge([mask, mask, mask])

    composited = (warped & mask3) + (composited & (~mask3))

else:
    # Fallback: use bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)
    # Resize image to fit within (w,h) preserving aspect ratio
    src_h, src_w = image_bgr.shape[:2]
    scale = min(w/src_w, h/src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Center within the rect
    x_off = x + (w - new_w)//2
    y_off = y + (h - new_h)//2
    composited[y_off:y_off+new_h, x_off:x_off+new_w] = resized

# Save output
cv2.imwrite(output_path, composited)

# Return result preview size
Image.open(output_path).resize((768, 768))
