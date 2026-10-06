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


def crop_document(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image load nahi hui: {image_path}")

    h, w = img.shape[:2]
    orig = img.copy()

    contour = detect_page_contour(img)

    if contour is None:
        cv2.imwrite(output_path, orig)
        return output_path, orig

    ratio = img.shape[0] / 800.0
    pts = contour.reshape(4, 2) * ratio
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    contour_area = cv2.contourArea(contour) * (ratio ** 2)
    coverage = contour_area / (h * w)

    if coverage < 0.35:
        cv2.imwrite(output_path, orig)
        return output_path, orig

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    aspect = maxHeight / maxWidth
    if not (0.6 < aspect < 1.7):
        cv2.imwrite(output_path, orig)
        return output_path, orig

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight),
                                  flags=cv2.INTER_CUBIC)

    cv2.imwrite(output_path, warped)
    return output_path, warped
