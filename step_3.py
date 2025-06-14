import cv2

class Step3Processor:
    def __init__(self, min_radius=5, max_radius=100):
        self.min_area = 3.14 * (min_radius ** 2) * 1.5  # Increased min_area
        self.max_area = 3.14 * (max_radius ** 2)

    def process(self, binary_img, original_img=None):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_img = original_img.copy() if original_img is not None else cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        colony_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Additional filtering based on contour perimeter
            perimeter = cv2.arcLength(cnt, True)
            if self.min_area < area < self.max_area and perimeter > 30:  # Add perimeter check
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(result_img, center, radius, (0, 255, 0), 2)
                colony_count += 1

        return result_img, colony_count

