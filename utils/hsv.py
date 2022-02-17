import cv2
import numpy as np

def get_hsv_masked(image):
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    # manually selected range for green leaves
    mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([90, 255, 255]))
    hsv[np.where(mask==0)] = 0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
