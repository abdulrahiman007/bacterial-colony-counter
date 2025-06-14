import cv2
import numpy as np

class Step2Processor:
    def __init__(self, method='adaptive', block_size=15, c=5):
        self.method = method
        self.block_size = block_size
        self.c = c

    def process(self, img):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        if self.method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                blurred,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=self.block_size,
                C=self.c
            )
        else:
            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close small holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove small noise

        return binary

