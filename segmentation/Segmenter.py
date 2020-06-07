import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Utils.plotting.plot_utils import plot_image


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
        under_low_mask = clean < self._low_threshold
        above_high_mask = self._high_threshold < clean
        middle_mask = ~np.logical_or(under_low_mask, above_high_mask)
        clean[under_low_mask] = 0
        clean[middle_mask] = 1
        clean[above_high_mask] = 2

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
        n_bins = 256
        threshold_factor = (256 // n_bins)
        hist, bins = np.histogram(image.flatten(), bins=n_bins)

        smooth_hist = hist
        smooth_hist = self._smooth_curve(smooth_hist, 20 // threshold_factor)
        smooth_hist = self._smooth_curve(smooth_hist, 14 // threshold_factor)
        smooth_hist = self._smooth_curve(smooth_hist, 10 // threshold_factor)
        smooth_hist = self._smooth_curve(smooth_hist, 6 // threshold_factor)

        # plt.figure()
        # plt.plot(hist, color="blue")
        # plt.plot(smooth_hist, color="red")
        # plt.show()

        mins, mins_inds = self._find_local_minima(smooth_hist)
        sorted_inds = sorted(filter(lambda x: 20 / threshold_factor < x < n_bins - 20 / threshold_factor, mins_inds))

        # assert len(sorted_inds) == self._num_classes - 1

        return sorted_inds[0] * threshold_factor, sorted_inds[1] * threshold_factor, hist, smooth_hist

    def segment_image_by_kmeans(self, image):
        clean = image
        clean = self._noise_cleaner.equalize_histogram(clean)
        clean = self._noise_cleaner.clean_salt_and_pepper(clean)

        segmentation_map = np.reshape(self._kmeans.fit_predict(clean.reshape(-1, 1)), clean.shape).astype(np.uint8)

        return segmentation_map

    def segment_image_by_threshold(self, image):
        clean = image.copy()

        clean = self._noise_cleaner.equalize_histogram(clean)
        clean = self._noise_cleaner.clean_salt_and_pepper(clean)

        hist, smooth_hist = None, None
        if self._auto_thresholds:
            low_thres, high_thres, hist, smooth_hist = self._infer_thresholds_by_histogram(clean)
            # I allow the high thres to go down, and the low to go up, but not vice versa.
            # self._low_threshold, self._high_threshold = max(low_thres, self._low_threshold), min(high_thres, self._high_threshold)
            self._low_threshold, self._high_threshold = low_thres, high_thres

        segmentation_map = self._perform_thresholding(clean)
        return segmentation_map, hist, smooth_hist, self._low_threshold, self._high_threshold

    def infer_region_statistics(self, image, mask):
        """
        This was supposed to be used for determining 3 regions in the warped image, then calculate the color
        distribution within each segment, then know for each pixel its probability of being an outlier by color.
        Unfortunately, I didn't have the time to create a good enough segmentation per class and this was no good.
        """
        segment_image = self.segment_image_by_kmeans(image.astype('uint8'))
        segment_image[~mask] = self._config.segmentation.num_classes

        statistics_per_class = []
        for c in range(self._config.segmentation.num_classes):
            class_data = image[segment_image == c]
            m, s = class_data.mean(), class_data.std()
            statistics_per_class.append((m, s))

        plot_image(image, "image")
        plot_image(segment_image, "segment_image")
        return statistics_per_class, segment_image


