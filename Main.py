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


def detect(inspected, noise_cleaner, warp_mask, warped, diff, warped_segmented, statistics_per_class_sorted):
    blured_diff_segmenter = BluredDiffSegmenter()
    low_diff_far_from_edge_segmenter = LowDiffFarFromEdgeSegmenter()

    blured_diff_seg_mask = blured_diff_segmenter.detect(inspected, warped, warp_mask)
    low_diff_far_from_edge_seg_mask = low_diff_far_from_edge_segmenter.detect(inspected, warped, warp_mask)

    total_defect_mask = np.logical_or(blured_diff_seg_mask, low_diff_far_from_edge_seg_mask)

    plot_image(total_defect_mask, "total_defect_mask")

    return total_defect_mask


def clean_false_positives(config, dirty_defect_mask, inspected, warped, warp_mask, diff, noise_cleaner, warped_segmented, statistics_per_class):
    dirty_defect_mask_dilated = noise_cleaner.dilate(dirty_defect_mask.astype('uint8'), diameter=5).astype(np.bool)  # in case of misses


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

    diff_above_thres_mask = 28 < diff
    clean_defect_mask = np.zeros_like(warp_mask)
    clean_defect_mask[dirty_defect_mask_dilated > 0] = True

    clean_defect_mask = np.logical_and(diff_above_thres_mask, clean_defect_mask)

    plot_image(inspected, "inspected")

    # plot_image(diff, "diff")
    plot_image(dirty_defect_mask, "dirty_defect_mask")
    # plot_image(diff_above_thres_mask, "diff_above_thres_mask")
    plot_image(clean_defect_mask, "clean_defect_mask")
    return clean_defect_mask


if __name__ == "__main__":
    def main():
        """
         This is just a mockup of the notebook report
        """
        from Utils.ConfigProvider import ConfigProvider
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        config = ConfigProvider.config()
        plt.close('all')

        # read data
        # inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
        # reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')
        inspected = cv2.imread(config.data.defective_inspected_path2, 0).astype('float32')
        reference = cv2.imread(config.data.defective_reference_path2, 0).astype('float32')
        # inspected = cv2.imread(config.data.non_defective_inspected_path, 0).astype('float32')
        # reference = cv2.imread(config.data.non_defective_reference_path, 0).astype('float32')

        # clean noise
        noise_cleaner = NoiseCleaner()

        inspected_clean = noise_cleaner.clean_salt_and_pepper(inspected, 5)
        reference_clean = noise_cleaner.clean_salt_and_pepper(reference, 5)

        inspected_eq = noise_cleaner.equalize_histogram(inspected_clean.astype('uint8'))
        reference_eq = noise_cleaner.equalize_histogram(reference_clean.astype('uint8'))





        # registration
        aligner = Aligner()
        resize = 5  # subpixel accuracy resolution
        moving_should_be_strided_by_10 = aligner.align_using_normxcorr(static=cv2.resize(inspected_eq,
                                                                                         (0, 0),
                                                                                         fx=resize,
                                                                                         fy=resize),
                                                                       moving=cv2.resize(reference_eq,
                                                                                         (0, 0),
                                                                                         fx=resize,
                                                                                         fy=resize))
        moving_should_be_strided_by = np.array(moving_should_be_strided_by_10) / resize

        warped, warp_mask = aligner.align_using_shift(inspected, reference, moving_should_be_strided_by)
        plot_image(warped, "warped")
        # plt.show()

        diff = np.zeros(inspected.shape, dtype=np.float32)
        diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]
        # diff[~warp_mask] = 0
        # also get rid of registration inaccuracy on the frame
        frame_radius = 3
        diff[noise_cleaner.dilate((~warp_mask).astype('uint8'), frame_radius) > 0] = 0
        # plot_image(diff.astype('uint8'), "diff")

        segmented_image = reference
        # statistics_per_class, _ = segment(reference, config)
        statistics_per_class, warped_segmented = segment(warped, config, warp_mask)

        # statistics_per_class_sorted = sorted(statistics_per_class, key=lambda item: item[0])
        # # image_statistics_sorted = sorted(statistics_per_class, key=lambda item: item[0])
        # for i, (m, s) in statistics_per_class_sorted:

        dirty_defect_mask = detect(inspected, noise_cleaner, warp_mask, warped, diff, warped_segmented, statistics_per_class)
        clean_false_positives(config, dirty_defect_mask, inspected, warped, warp_mask, diff, noise_cleaner, warped_segmented, statistics_per_class)

        plt.show()
    main()
