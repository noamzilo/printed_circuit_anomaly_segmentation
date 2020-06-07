from Utils.ConfigProvider import ConfigProvider
from Utils.plotting.plot_utils import plot_image
from Utils.plotting.plot_utils import show_color_diff
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
from defect_segmentation.BaseSegmenter import BaseSegmenter
from overrides import overrides


class DiffSegmenter(BaseSegmenter):
    """
    simplest segmenter, by using a high threshold for noise
    """
    def __init__(self):
        self._config = ConfigProvider.config()
        self._noise_cleaner = NoiseCleaner()

        self._diff_thres = self._config.detection.diff_thres

    @overrides
    def detect(self, inspected, warped, warp_mask):
        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]

        diff = self._noise_cleaner.clean_frame(diff, warp_mask)

        detection_mask = self._diff_thres < diff

        # plots
        plot_image(diff, "diff")
        show_color_diff(warped, inspected, "color diff")
        plot_image(detection_mask, "diff_based_segmentation")

        return detection_mask

