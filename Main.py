from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.misc
from Utils.plotting.plot_utils import show_color_diff
from Utils.plotting.plot_utils import plot_image
from Utils.plotting.plot_utils import plot_image_3d
from alignment.Aligner import Aligner
from noise_cleaning.NoiseCleaner import NoiseCleaner
from segmentation.Segmenter import Segmenter

from defect_segmentation.BluredDiffSegmenter import BluredDiffSegmenter
from defect_segmentation.LowDiffFarFromEdgeSegmenter import LowDiffFarFromEdgeSegmenter
from defect_segmentation.DefectSegmentationRefineer import DefectSegmentationRefiner


def segment(image, config, mask):
    plot_image(image, "image")
    segmenter = Segmenter()
    segment_image = segmenter.segment_image_by_kmeans(image.astype('uint8'))
    segment_image[~mask] = config.segmentation.num_classes
    plot_image(segment_image, "segment_image")

    statistics_per_class = []
    for c in range(config.segmentation.num_classes):
        class_data = image[segment_image == c]
        m, s = class_data.mean(), class_data.std()
        statistics_per_class.append((m, s))
    return statistics_per_class, segment_image


def detect_defects(inspected, warp_mask, warped):
    blured_diff_segmenter = BluredDiffSegmenter()
    low_diff_far_from_edge_segmenter = LowDiffFarFromEdgeSegmenter()

    blured_diff_seg_mask = blured_diff_segmenter.detect(inspected, warped, warp_mask)
    low_diff_far_from_edge_seg_mask = low_diff_far_from_edge_segmenter.detect(inspected, warped, warp_mask)

    total_defect_mask = np.logical_or(blured_diff_seg_mask, low_diff_far_from_edge_seg_mask)

    plot_image(total_defect_mask, "total_defect_mask")

    return total_defect_mask


if __name__ == "__main__":
    def main():
        config = ConfigProvider.config()

        # read data
        # inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
        # reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')
        inspected = cv2.imread(config.data.defective_inspected_path2, 0).astype('float32')
        reference = cv2.imread(config.data.defective_reference_path2, 0).astype('float32')
        # inspected = cv2.imread(config.data.non_defective_inspected_path, 0).astype('float32')
        # reference = cv2.imread(config.data.non_defective_reference_path, 0).astype('float32')

        # registration
        aligner = Aligner()
        warped, warp_mask = aligner.align_images(static=inspected, moving=reference)


        dirty_defect_mask = detect_defects(inspected, warp_mask, warped)

        refiner = DefectSegmentationRefiner()
        segmentation_result = refiner.refine_segmentation(dirty_defect_mask, inspected, warped, warp_mask)

        plt.show()
    main()
