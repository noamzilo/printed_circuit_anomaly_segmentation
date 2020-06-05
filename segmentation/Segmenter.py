import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner


class Segmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._num_classes = self._config.segmentaion.num_classes
        self._low_threshold = self._config.noise_cleaning.low_threshold
        self._high_threshold = self._config.noise_cleaning.high_threshold

        self._noise_cleaner = NoiseCleaner()

    def _perform_thresholding(self, image):
        clean = image.copy()
        clean[clean < self._low_threshold] = 0
        clean[np.logical_and(0 < clean, clean < self._high_threshold)] = 125
        clean[self._high_threshold < clean] = 255

        return clean

    def segment_image(self, image):
        thresholded = self._perform_thresholding(image)
        clean = self._noise_cleaner.clean_salt_and_pepper(thresholded)
        r_edges = cv2.Canny(clean, 100, 200)

