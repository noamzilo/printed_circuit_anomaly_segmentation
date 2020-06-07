from Utils.ConfigProvider import ConfigProvider
from Utils.plotting.plot_utils import plot_image
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner


class DefectSegmentationRefiner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._noise_cleaner = NoiseCleaner()
        self._frame_radius = self._config.detection.frame_radius

        self._min_diff_threshold = self._config.detection.min_diff_threshold

    def refine_segmentation(self, dirty_defect_mask, inspected, warped, warp_mask):
        # I wish this had worked. need better segmentation or more tricks.
        # max_value_per_pixel = np.zeros_like(diff)
        # min_value_per_pixel = np.zeros_like(diff)
        # for c in range(config.segmentation.num_classes):
        #     max_value_per_pixel[warped_segmented == c] = statistics_per_class[c][0] + 2 * statistics_per_class[c][1]
        #     min_value_per_pixel[warped_segmented == c] = statistics_per_class[c][0] - 2 * statistics_per_class[c][1]
        #
        # clean_defect_mask = np.zeros_like(dirty_defect_mask)
        # clean_defect_mask[dirty_defect_mask] = max_value_per_pixel[dirty_defect_mask] < inspected[dirty_defect_mask]
        # clean_defect_mask[dirty_defect_mask] = np.logical_or(clean_defect_mask[dirty_defect_mask], min_value_per_pixel[dirty_defect_mask] < inspected[dirty_defect_mask])
        # plot_image(max_value_per_pixel, "max_value_per_pixel")
        # plot_image(min_value_per_pixel, "min_value_per_pixel")

        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]

        # get rid of registration inaccuracy on the frame
        diff[self._noise_cleaner.dilate((~warp_mask).astype('uint8'), self._frame_radius) > 0] = 0

        # enlarge detection area in case of close proximity misses
        dirty_defect_mask_dilated = \
            self._noise_cleaner.dilate(dirty_defect_mask.astype('uint8'), diameter=5).astype(np.bool)

        diff_above_thres_mask = self._min_diff_threshold < diff

        clean_defect_mask = np.logical_and(diff_above_thres_mask, dirty_defect_mask_dilated)

        plot_image(inspected, "inspected")
        plot_image(dirty_defect_mask, "dirty_defect_mask")
        plot_image(clean_defect_mask, "clean_defect_mask")
        return clean_defect_mask
