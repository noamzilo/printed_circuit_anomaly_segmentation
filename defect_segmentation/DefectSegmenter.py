from Utils.ConfigProvider import ConfigProvider
from defect_segmentation.BluredDiffSegmenter import BluredDiffSegmenter
from defect_segmentation.LowDiffFarFromEdgeSegmenter import LowDiffFarFromEdgeSegmenter
from defect_segmentation.ThreadDefectSegmenter import ThreadDefectSegmenter
from defect_segmentation.DefectSegmentationRefineer import DefectSegmentationRefiner

import numpy as np
from Utils.plotting.plot_utils import plot_image


class DefectSegmenter(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._refiner = DefectSegmentationRefiner()
        self._blured_diff_segmenter = BluredDiffSegmenter()
        self._low_diff_far_from_edge_segmenter = LowDiffFarFromEdgeSegmenter()
        self._thread_defect_segmenter = ThreadDefectSegmenter()

    def segment_defects(self, inspected, warped, warp_mask):
        dirty_defect_mask = self._detect_defects(inspected, warp_mask, warped)
        segmentation_result = self._refiner.refine_segmentation(dirty_defect_mask, inspected, warped, warp_mask)
        return segmentation_result

    def _detect_defects(self, inspected, warp_mask, warped):
        blured_diff_seg_mask = self._blured_diff_segmenter.detect(inspected, warped, warp_mask)
        low_diff_far_from_edge_seg_mask = self._low_diff_far_from_edge_segmenter.detect(inspected, warped, warp_mask)
        thread_defect_seg_mask = self._thread_defect_segmenter.detect(inspected, warped, warp_mask)

        total_defect_mask = np.logical_or(blured_diff_seg_mask, low_diff_far_from_edge_seg_mask)
        total_defect_mask = np.logical_or(total_defect_mask, thread_defect_seg_mask)

        plot_image(total_defect_mask, "total_defect_mask")
        return total_defect_mask
