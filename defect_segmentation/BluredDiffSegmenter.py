from Utils.ConfigProvider import ConfigProvider
from Utils.plotting.plot_utils import plot_image
from Utils.plotting.plot_utils import show_color_diff
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner


class BluredDiffSegmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._noise_cleaner = NoiseCleaner()

        self._frame_radius = self._config.detection.frame_radius
        self._blured_diff_thres = self._config.detection.blured_diff_thres

    def detect(self, inspected, warped, warp_mask):
        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]

        # get rid of registration inaccuracy on the frame
        diff[self._noise_cleaner.dilate((~warp_mask).astype('uint8'), self._frame_radius) > 0] = 0

        # make sure noise is not interfering
        diff_blured = self._noise_cleaner.blur(diff, sigma=7)
        detection_mask = self._blured_diff_thres < diff_blured

        # plots
        plot_image(diff, "diff")
        show_color_diff(warped, inspected, "color diff")
        plot_image(diff_blured, "diff_blured")
        plot_image(detection_mask, "diff_based_segmentation")

        return detection_mask

