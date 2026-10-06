import cv2
import numpy as np
import imutils
from .utils import order_points


def detect_page_contour(image):
    ratio = image.shape[0] / 800.0
    small = imutils.resize(image, height=800)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged1 = cv2.Canny(gray, 30, 100)
    edged2 = cv2.Canny(gray, 75, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edged1 = cv2.morphologyEx(edged1, cv2.MORPH_CLOSE, kernel)
    edged2 = cv2.morphologyEx(edged2, cv2.MORPH_CLOSE, kernel)

    candidates = []
    img_area = small.shape[0] * small.shape[1]

    for edged in (edged1, edged2):
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.20 * img_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append((area, approx))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def trim_extra_space(image, threshold=15, padding=20):
    """
    Image ke around jo extra white/black space hai use hata deta hai.
    
    Args:
        image: OpenCV image (BGR or grayscale)
        threshold: Kitna difference blank maana jaye (0-255)
        padding: Content ke around kitna margin chhodna hai
    
    Returns:
        Trimmed image
    """
    if image is None or image.size == 0:
        return image
    
    # Grayscale me convert karo (agar color hai to)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Blur karo — noise hatao
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect karo kaunse pixels "content" hain aur kaunse "background"
    # Background = jo bhi sabse common value hai (white ya black)
    # Histogram se pata karo
    hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # Sabse common intensity = background
    bg_intensity = int(np.argmax(hist))
    
    # Content pixels = jo background se alag hain
    diff = np.abs(blurred.astype(int) - bg_intensity)
    
    # Binary mask — content = 255, background = 0
    content_mask = (diff > threshold).astype(np.uint8) * 255
    
    # Morphological operations — chhote noise dots hatao
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel)
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, kernel)
    
    # Content ke coordinates dhoondo (bounding box)
    coords = cv2.findNonZero(content_mask)
    
    if coords is None:
        # Kuch bhi content nahi mila — original return karo
        return image
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Padding add karo (content ke around thoda margin)
    img_h, img_w = gray.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    
    # Crop karo
    trimmed = image[y1:y2, x1:x2]
    
    # Agar crop bahut chhota ho gaya (95% se zyada hat gaya), to original use karo
    area_ratio = (trimmed.shape[0] * trimmed.shape[1]) / (img_h * img_w)
    if area_ratio < 0.05:
        return image
    
    return trimmed


def crop_document(image_path, output_path):
    """
    Document page auto-crop + extra blank space trim.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image load nahi hui: {image_path}")

    h, w = img.shape[:2]
    orig = img.copy()

    contour = detect_page_contour(img)

    if contour is None:
        # Page detect nahi hua — sirf blank space trim karo
        trimmed = trim_extra_space(orig, threshold=15, padding=20)
        cv2.imwrite(output_path, trimmed)
        return output_path, trimmed

    ratio = img.shape[0] / 800.0
    pts = contour.reshape(4, 2) * ratio
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    contour_area = cv2.contourArea(contour) * (ratio ** 2)
    coverage = contour_area / (h * w)

    if coverage < 0.35:
        # Coverage kam hai — sirf blank space trim karo
        trimmed = trim_extra_space(orig, threshold=15, padding=20)
        cv2.imwrite(output_path, trimmed)
        return output_path, trimmed

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    aspect = maxHeight / maxWidth
    if not (0.6 < aspect < 1.7):
        trimmed = trim_extra_space(orig, threshold=15, padding=20)
        cv2.imwrite(output_path, trimmed)
        return output_path, trimmed

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight),
                                  flags=cv2.INTER_CUBIC)

    # ===== NAYA STEP: Extra blank space trim karo =====
    warped = trim_extra_space(warped, threshold=15, padding=20)

    cv2.imwrite(output_path, warped)
    return output_path, warped
