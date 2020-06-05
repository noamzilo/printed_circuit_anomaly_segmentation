import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np


class NoiseCleaner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._low_threshold = self._config.noise_cleaning.low_threshold
        self._high_threshold = self._config.noise_cleaning.high_threshold
        self._gaussian_blur_radius = self._config.noise_cleaning.gaussian_blur_radius
        self._median_blur_radius = self._config.noise_cleaning.median_blur_radius
        self._erode_dilate_diameter = self._config.noise_cleaning.erode_dilate_diameter

    def perform_thresholding(self, image):
        clean = image.copy()
        clean[clean < self._low_threshold] = 0
        clean[np.logical_and(0 < clean, clean < self._high_threshold)] = 125
        clean[self._high_threshold < clean] = 255

        return clean

    def clean_salt_and_pepper(self, image):
        clean = cv2.medianBlur(image, self._median_blur_radius)
        return clean

    def blur(self, image):
        clean = cv2.blur(image, (self._gaussian_blur_radius, self._gaussian_blur_radius))
        return clean

    def open(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self._erode_dilate_diameter, self._erode_dilate_diameter))
        clean = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)
        return clean

    def close(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self._erode_dilate_diameter, self._erode_dilate_diameter))
        clean = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
        return clean
