import cv2
import numpy as np

class Step1Processor:
    def __init__(self, min_radius):
        self.old_img = None
        self.old_min_rad = min_radius
        self.kernel_size = self._compute_kernel_size()

    def _compute_kernel_size(self):
        if self.old_img is None:
            return 0
        s = min(self.old_img.shape[1] // 3, self.old_img.shape[0] // 3)
        self.old_min_rad = min(self.old_min_rad, s)
        return (((self.old_min_rad - 1) // 4) * 2) + 1 if self.old_min_rad > 1 else 0

    def update_params(self, src_img):
        self.old_img = src_img.copy()
        self.kernel_size = self._compute_kernel_size()

    def need_reprocess(self, src_img, min_radius):
        if self.old_img is None:
            return True
        if not np.array_equal(self.old_img, src_img):
            return True
        if self.old_min_rad != min_radius:
            return True
        return False

    def process(self, src_img):
        # Step 1: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(src_img)
        
        # Step 2: Noise reduction with median blur
        median_img = cv2.medianBlur(clahe_img, 5)  # Increased kernel size for better noise reduction
        
        # Step 3: Apply Gaussian blur
        ksize = max(3, (self.old_min_rad // 2) | 1)
        processed = cv2.GaussianBlur(median_img, (ksize, ksize), 0)
        
        return processed
