import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Segmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._num_classes = self._config.segmentation.num_classes
        self._low_threshold = self._config.segmentation.low_threshold
        self._high_threshold = self._config.segmentation.high_threshold
        self._auto_thresholds = self._config.segmentation.auto_thresholds

        self._noise_cleaner = NoiseCleaner()
        self._kmeans = KMeans(n_clusters=self._num_classes)

    def _perform_thresholding(self, image):
        clean = image.copy()
        clean[clean < self._low_threshold] = 0
        clean[np.logical_and(0 < clean, clean < self._high_threshold)] = 125
        clean[self._high_threshold < clean] = 255

        return clean

    @staticmethod
    def _find_local_minima( a):
        mins_mask = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
        mins_inds = np.where(mins_mask)[0]
        mins = a[mins_inds]
        return mins, mins_inds

    @staticmethod
    def _smooth_curve(curve, kernel_size=5):
        kernel = np.ones((kernel_size,)) / kernel_size
        smooth = np.convolve(curve, kernel, mode='same')
        return smooth

    def _infer_thresholds_by_histogram(self, image):
        """
        assuming 3 intensity levels, and a valid input image (non defective)
        :return: low and high thresholds by which the image can be roughly segmented.
        """
        clean = image
        clean = self._noise_cleaner.equalize_histogram(clean)
        clean = self._noise_cleaner.clean_salt_and_pepper(clean)

        n_bins = 128
        hist, bins = np.histogram(clean.flatten(), bins=n_bins)

        smooth_hist = hist
        smooth_hist = self._smooth_curve(smooth_hist, 10)
        smooth_hist = self._smooth_curve(smooth_hist, 7)
        smooth_hist = self._smooth_curve(smooth_hist, 5)
        smooth_hist = self._smooth_curve(smooth_hist, 3)

        plt.figure()
        plt.plot(hist, color="blue")
        plt.plot(smooth_hist, color="red")
        plt.show()

        mins, mins_inds = self._find_local_minima(smooth_hist)
        sorted_inds = sorted(filter(lambda x: 10 < x < n_bins - 10, mins_inds))

        assert len(sorted_inds) == self._num_classes - 1

        return sorted_inds[0], sorted_inds[1]

    def _infer_thresholds_by_knn(self, image):
        clean = image
        clean = self._noise_cleaner.equalize_histogram(clean)
        clean = self._noise_cleaner.clean_salt_and_pepper(clean)

        self._kmeans.fit(clean)
        asgsga

    def segment_image_by_threshold(self, image):
        if self._auto_thresholds:
            low_threshold, high_threshold = self._infer_thresholds_by_histogram(image)
        else:
            low_threshold, high_threshold = self._low_threshold, self._high_threshold
        # we now know
        thresholded = self._perform_thresholding(image)

        clean = self._noise_cleaner.clean_salt_and_pepper(thresholded)
        r_edges = cv2.Canny(clean, 100, 200)

_infer_thresholds_by_knn