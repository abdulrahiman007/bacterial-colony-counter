from app.logic.step_1 import Step1Processor
from app.logic.step_2 import Step2Processor
from app.logic.step_3 import Step3Processor

class ColonyProcessor:
    def __init__(self, min_radius):
        self.min_radius = min_radius

    def run(self, img):
        img1 = Step1Processor(self.min_radius).process(img)
        img2 = Step2Processor().process(img1)
        result_img, count = Step3Processor(self.min_radius).process(img2, img)
        return result_img, count
