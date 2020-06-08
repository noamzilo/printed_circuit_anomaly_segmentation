from Utils.ConfigProvider import ConfigProvider
import cv2
from matplotlib import pyplot as plt
import matplotlib
from Utils.plotting.plot_utils import show_color_diff, get_color_diff_image
from Utils.plotting.plot_utils import plot_image
from alignment.Aligner import Aligner
import numpy as np
from noise_cleaning.NoiseCleaner import NoiseCleaner
from defect_segmentation.DefectSegmenter import DefectSegmenter


def main():
    print("started")
    config = ConfigProvider.config()

    # read data
    inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
    reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')

    # registration
    aligner = Aligner()
    warped, warp_mask = aligner.align_images(static=inspected, moving=reference)

    # find defects
    defect_segmenter = DefectSegmenter()
    defect_mask = defect_segmenter.segment_defects(inspected, warped, warp_mask)

    # observe results
    diff = np.zeros(inspected.shape, dtype=np.float32)
    diff[warp_mask] = (np.abs((np.float32(warped) - np.float32(inspected))))[warp_mask]
    noise_cleaner = NoiseCleaner()
    diff = noise_cleaner.clean_frame(diff, warp_mask)

    cv2.imshow("color_result", get_color_diff_image(inspected, defect_mask * 255).astype('uint8'))
    plt.imshow(diff.astype('uint8'), cmap='gray')
    plt.title("diff")
    cv2.imshow("inspected", inspected.astype('uint8'))
    cv2.imshow("reference", reference.astype('uint8'))
    cv2.imshow("result", defect_mask.astype('uint8') * 255)

    plt.show()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
