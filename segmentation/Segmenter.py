import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner


class Segmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._num_classes = self._config.segmentaion.num_classes

        self._noise_cleaner = NoiseCleaner()

    def segment_image(self, image):
        thresholded = self._noise_cleaner.perform_thresholding(image)
        clean = self._noise_cleaner.clean_salt_and_pepper(thresholded)
        r_edges = cv2.Canny(clean, 100, 200)

