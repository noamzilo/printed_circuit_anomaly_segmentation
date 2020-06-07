from Utils.ConfigProvider import ConfigProvider
from Utils.plotting.plot_utils import plot_image
from Utils.plotting.plot_utils import show_color_diff
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
import cv2


class LowDiffFarFromEdgeSegmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._noise_cleaner = NoiseCleaner()

        self._aura_radius = self._config.detection.aura_radius
        self._low_diff_far_from_edge_thres = self._config.detection.low_diff_far_from_edge_thres

    def detect(self, inspected, warped, warp_mask):
        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]

        diff = self._noise_cleaner.clean_frame(diff, warp_mask)

        # find and dilate edges, to get rid of high diff caused by suboptimal registration
        # and is most apparent near real edges.
        edges = cv2.Canny(warped.astype('uint8'), 100, 200) > 0
        edges_dialated = self._noise_cleaner.dilate(edges.astype(np.float32), self._aura_radius)
        diff_no_edges = diff.copy()

        # blur to avoid noise
        diff_no_edges_blured = self._noise_cleaner.blur(diff_no_edges, sigma=5)
        #disregard area of edges
        diff_no_edges_blured[edges_dialated > 0] = 0
        low_diff_far_from_edge_mask = self._low_diff_far_from_edge_thres < diff_no_edges_blured


        plot_image(edges, "edges")
        plot_image(edges_dialated, "edges_dilated")
        plot_image(diff_no_edges_blured, "diff_no_edges_blured")
        plot_image(low_diff_far_from_edge_mask, "low_diff_far_from_edge_mask")
        return low_diff_far_from_edge_mask

