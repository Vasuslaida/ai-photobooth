import cv2
import numpy as np

def apply_filter(frame, mode):
    if mode == "gray":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif mode == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)

    elif mode == "red":
        frame[:, :, 0] = 0
        frame[:, :, 1] = 0
        return frame

    elif mode == "cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 9
        )
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)
    return frame