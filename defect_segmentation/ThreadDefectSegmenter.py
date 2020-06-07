from Utils.ConfigProvider import ConfigProvider
from Utils.plotting.plot_utils import plot_image
from Utils.plotting.plot_utils import show_color_diff
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
import cv2
import matplotlib.pyplot as plt


class ThreadDefectSegmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._noise_cleaner = NoiseCleaner()

        self._thread_defect_thres = self._config.detection.thread_defect_thres
        self._aura_radius = self._config.detection.aura_radius
        self._low_diff_far_from_edge_thres = self._config.detection.low_diff_far_from_edge_thres

        # params = cv2.SimpleBlobDetector_Params()
        # params.minThreshold = 100
        # params.maxThreshold = 5000
        #
        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 200
        #
        # # Filter by Circularity
        # params.filterByCircularity = False
        # params.minCircularity = 0.785
        #
        # # Filter by Convexity
        # params.filterByConvexity = False
        # params.minConvexity = 0.87
        # self._blob_detector = cv2.SimpleBlobDetector(params)

    def detect(self, inspected, warped, warp_mask):
        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]
        diff = self._noise_cleaner.clean_frame(diff, warp_mask)

        i_blured = self._noise_cleaner.blur(inspected, sigma=10)
        high_pass = np.abs(np.float32(i_blured) - np.float32(inspected))
        show_color_diff(i_blured, inspected, "high_pass")

        # find and dilate edges, to get rid of high diff in high pass image caused by real edges
        # this will leave us only with edges caused by stuff that weren't in the original image.
        # obvious limitation: won't find items near real edges in original image.
        edges = cv2.Canny(warped.astype('uint8'), 100, 200) > 0
        edges_dialated = self._noise_cleaner.dilate(edges.astype(np.float32), self._aura_radius)
        # blur to avoid noise
        high_pass_no_real_edges = high_pass.copy()
        high_pass_no_real_edges[edges_dialated > 0] = 0
        plot_image(high_pass_no_real_edges, "high_pass_no_real_edges")

        thread_defect_mask_noisy = self._thread_defect_thres < high_pass_no_real_edges

        thread_defect_mask_dilated = self._noise_cleaner.dilate(thread_defect_mask_noisy.astype(np.float32), diameter=10, iterations=1)
        # thread_defect_mask_clean = self._noise_cleaner.open(thread_defect_mask_noisy.astype(np.float32), diameter=1, iterations=1)

        # blob_points = self._blob_detector.detect(thread_defect_mask_dilated)
        # im_with_keypoints = cv2.drawKeypoints(np.zeros_like(thread_defect_mask_dilated), blob_points, np.array([]),
        #                                       255, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plot_image(thread_defect_mask_noisy, "thread_defect_mask_noisy")
        plot_image(thread_defect_mask_dilated, "thread_defect_mask_dilated")
        # plot_image(im_with_keypoints, "im_with_keypoints")
        return thread_defect_mask_noisy

