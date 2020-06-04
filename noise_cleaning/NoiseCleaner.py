import cv2
from Utils.ConfigProvider import ConfigProvider


class NoiseCleaner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._gaussian_blur_radius = self._config.noise_cleaning.gaussian_blur_radius
        self._median_blur_radius = self._config.noise_cleaning.median_blur_radius

    def clean_noise(self, image):
        blured = cv2.medianBlur(image, self._median_blur_radius)
        blured = cv2.blur(blured,
                          (self._gaussian_blur_radius, self._gaussian_blur_radius))

        return blured
