import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np
from skimage.filters.rank import majority
from skimage.morphology import square
from skimage.restoration import denoise_bilateral


class NoiseCleaner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._median_blur_radius = self._config.noise_cleaning.median_blur_radius
        self._frame_radius = self._config.noise_cleaning.frame_radius

    def clean_salt_and_pepper(self, image, radius=None):
        if radius is None:
            clean = cv2.medianBlur(image, self._median_blur_radius)
        else:
            clean = cv2.medianBlur(image, radius)
        return clean

    def blur(self, image, sigma=5):
        clean = cv2.blur(image, (sigma, sigma))
        return clean

    def open(self, image, diameter=3, iterations=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (diameter, diameter))
        clean = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return clean

    def dilate(self, image, diameter, iterations=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (diameter, diameter))
        clean = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=iterations)
        return clean

    def erode(self, image, diameter, iterations=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (diameter, diameter))
        clean = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=iterations)
        return clean

    def close(self, image, diameter=5, iterations=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (diameter, diameter))
        clean = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return clean

    @staticmethod
    def equalize_histogram(image):
        return cv2.equalizeHist(image)

    def majority(self, image, radius):
        return majority(image.astype('uint8'), square(2 * radius + 1))

    def bilateral_filter(self, image, sigma_color=0.05, sigma_spatial=15):
        return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=False)

    def clean_frame(self, image, warp_mask):
        # get rid of registration inaccuracy on the frame
        image[self.dilate((~warp_mask).astype('uint8'), self._frame_radius) > 0] = 0
        return image

    def clean_stray_pixels_bw(self, bw_image, min_size):
        thread_defect_mask_clean = bw_image.copy()
        ret, connected_components_labels = cv2.connectedComponents(bw_image.astype('uint8'), connectivity=8)
        for label in range(1, ret):
            label_count = np.count_nonzero(label == connected_components_labels)
            if label_count < min_size:
                thread_defect_mask_clean[label == connected_components_labels] = 0
        return thread_defect_mask_clean
