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

# TODO can get dark/light on gray by simple thresholding on diff_blured


def detect_on_gray_areas(inspected, noise_cleaner, warp_mask, warped, diff):
    inspected_eq = noise_cleaner.equalize_histogram(inspected.astype('uint8').copy())
    warped_eq = np.ones_like(warped) * 128  # to not destroy the hist_eq
    warped_eq[warp_mask] = noise_cleaner.equalize_histogram(warped.astype('uint8'))[warp_mask]
    plot_image(inspected_eq, "inspected_eq")
    plot_image(warped_eq, "warped_eq")

    diff_eq = np.zeros(inspected.shape, dtype=np.float32)
    diff_eq[warp_mask] = (np.abs((np.float32(warped_eq) - np.float32(inspected_eq))))[warp_mask]

    plot_image(diff_eq, "diff_eq")

    sigma = 10
    # diff_blured = noise_cleaner.blur(diff.copy(), sigma=sigma)
    i_blured = noise_cleaner.clean_salt_and_pepper(np.float32(inspected_eq), radius=5)
    w_blured = noise_cleaner.clean_salt_and_pepper(np.float32(warped_eq), radius=5)
    diff_blured = np.abs(np.float32(i_blured) - np.float32(w_blured))
    plot_image(diff_blured, "diff_blured")

    #
    # diff_blured_above_thres_mask = 25 < diff_blured
    # diff_above_thres_mask = 33 < diff

    # find defects NOT on gray background
    show_color_diff(warped_eq, inspected_eq, "color diff")
    # diff_blured = noise_cleaner.blur(diff, sigma=7)  # 5 finds a false negative on non the defective image set
    plot_image(diff_blured, "diff_blured")
    high_defect_mask_diff_blured = 125 < diff_blured
    plot_image(high_defect_mask_diff_blured, "high_defect_mask_diff_blured")

    # find defects on gray background
    # edges = cv2.Canny(warped.astype('uint8'), 100, 200) > 0
    # glowy_radius = 5
    # edges_dialated = noise_cleaner.dilate(edges.astype(np.float32), glowy_radius)
    # diff_no_edges = diff_eq.copy()
    # diff_no_edges_blured = noise_cleaner.blur(diff_no_edges, sigma=3)
    # diff_no_edges_blured[edges_dialated > 0] = 0
    # plot_image(edges, "edges")
    # plot_image(edges_dialated, "edges_dilated")
    plot_image(diff_eq, "diff_eq")
    high_defect_mask = 25 < diff_eq

    # combine results
    plot_image(high_defect_mask, "high_defect_mask")
    # high_defect_mask_closure = noise_cleaner.close(high_defect_mask.astype('uint8'), diameter=20)
    # This will cause false positives if many nearby defects, but this isn't probable in the business domain.
    # plot_image(high_defect_mask_closure, "high_defect_mask_closure")

    # total_defect_mask = np.logical_or(high_defect_mask_diff_blured, high_defect_mask_closure)
    total_defect_mask = np.logical_or(high_defect_mask_diff_blured, high_defect_mask)
    plot_image(total_defect_mask, "total_defect_mask")

    # in case of in-defect-misses due to high threshold, which still caught some of the defect
    dialated_mask = noise_cleaner.dilate(total_defect_mask.astype('uint8'), diameter=8)
    plot_image(dialated_mask, "dialated_mask")

    return dialated_mask


def clean_false_positives(dirty_defect_mask, inspected, warped, warp_mask, noise_cleaner):
    sigma = 5
    # diff_blured = noise_cleaner.blur(diff.copy(), sigma=sigma)
    i_blured = noise_cleaner.blur(np.float32(inspected), sigma=sigma)
    w_blured = noise_cleaner.blur(np.float32(warped), sigma=sigma)
    diff_blured = np.abs(np.float32(i_blured) - np.float32(w_blured))
    diff = np.abs(np.float32(inspected) - np.float32(warped))

    diff_blured_above_thres_mask = 25 < diff_blured
    diff_above_thres_mask = 33 < diff

    clean_defect_mask = np.zeros_like(warp_mask)
    clean_defect_mask[dirty_defect_mask > 0] = True

    clean_defect_mask = np.logical_and(diff_blured_above_thres_mask, clean_defect_mask)

    plot_image(diff, "diff")
    plot_image(diff_blured, "diff_blured")
    plot_image(diff_above_thres_mask, "diff_above_thres_mask")
    plot_image(diff_blured_above_thres_mask, "diff_blured_above_thres_mask")
    plot_image(dirty_defect_mask, "dirty_defect_mask")
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

        inspected_clean = noise_cleaner.equalize_histogram(inspected_clean.astype('uint8').copy())
        reference_clean = noise_cleaner.equalize_histogram(reference_clean.astype('uint8').copy())

        # registration
        aligner = Aligner()
        resize = 5  # subpixel accuracy resolution
        moving_should_be_strided_by_10 = aligner.align_using_normxcorr(static=cv2.resize(inspected_clean,
                                                                                         (0, 0),
                                                                                         fx=resize,
                                                                                         fy=resize),
                                                                       moving=cv2.resize(reference_clean,
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
        plot_image(diff.astype('uint8'), "diff")

        # diff = noise_cleaner.equalize_histogram(diff.astype('uint8').copy())
        dirty_defect_mask = detect_on_gray_areas(inspected, noise_cleaner, warp_mask, warped, diff)
        clean_false_positives(dirty_defect_mask, inspected, warped, warp_mask, noise_cleaner)

        plt.show()
    main()
