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

if __name__ == "__main__":
    def main():
        """
         This is just a mockup of the notebook report
        """

        # imports
        from Utils.ConfigProvider import ConfigProvider
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from IPython.display import display
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        config = ConfigProvider.config()
        plt.close('all')

        from noise_cleaning.NoiseCleaner import NoiseCleaner
        noise_cleaner = NoiseCleaner()

        # read data
        inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
        reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')

        # clean noise
        inspected = noise_cleaner.clean_salt_and_pepper(inspected, 5)
        reference = noise_cleaner.clean_salt_and_pepper(reference, 5)

        # alignment
        from alignment.Aligner import Aligner
        aligner = Aligner()

        # registration
        resize = 5  # subpixel accuracy resolution
        moving_should_be_strided_by_10 = aligner.align_using_normxcorr(static=cv2.resize(inspected,
                                                                                         (0, 0),
                                                                                         fx=resize,
                                                                                         fy=resize),
                                                                       moving=cv2.resize(reference,
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
        # plt.show()

        show_color_diff(warped, inspected, "color diff")
        # plt.show()

        diff_blured = noise_cleaner.blur(diff, sigma=5)
        plot_image(diff_blured, "diff_blured")
        # plt.show()

        high_defect_thres = 30
        high_defect_mask_bad = high_defect_thres < diff_blured
        plot_image(high_defect_mask_bad, "high_defect_mask_bad")
        # plt.show()
        # this still leaves edges in as defects

        edges = cv2.Canny(warped.astype('uint8'), 100, 200) > 0
        glowy_radius = 5
        edges_dialated = noise_cleaner.dilate(edges.astype(np.float32), glowy_radius)
        diff_no_edges = diff.copy()
        diff_no_edges[edges_dialated > 0] = 0

        plot_image(edges, "edges")
        plot_image(edges_dialated, "edges_dilated")
        plot_image(diff_no_edges, "diff_no_edges")
        # plt.show()

        high_defect_mask = high_defect_thres < diff_no_edges
        plot_image(high_defect_mask, "high_defect_mask")

        high_defect_mask_closure = noise_cleaner.close(high_defect_mask.astype('uint8'), diameter=20)
        # this will cause false positives if many nearby defects, but this isn't probable in the business domain.
        plot_image(high_defect_mask_closure, "high_defect_mask_closure")

        plt.show()

        hi = 5


    main()
