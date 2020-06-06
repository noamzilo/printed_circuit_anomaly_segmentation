from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    def main():
        """
         This is just a mockup of the notebook report
        """

        # imports
        from Utils.ConfigProvider import ConfigProvider
        import os
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from IPython.display import display
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        config = ConfigProvider.config()
        plt.close('all')

        # read data
        inspected = cv2.imread(config.data.defective_inspected_path1, 0)
        reference = cv2.imread(config.data.defective_reference_path1, 0)
        diff = np.abs(inspected - reference)

        plt.figure()
        plt.title('inspected')
        plt.imshow(inspected, cmap='gray')
        plt.figure()
        plt.title('reference')
        plt.imshow(reference, cmap='gray')
        plt.figure()
        plt.title('diff')
        plt.imshow(diff, cmap='gray')

        # clean noise
        from noise_cleaning.NoiseCleaner import NoiseCleaner
        noise_cleaner = NoiseCleaner()
        r_blured = noise_cleaner.clean_salt_and_pepper(reference)
        i_blured = noise_cleaner.clean_salt_and_pepper(inspected)

        plt.figure()
        plt.title('r_blured')
        plt.imshow(r_blured, cmap='gray')
        # plt.figure()
        # plt.title('i_blured')
        # plt.imshow(i_blured, cmap='gray')
        # plt.figure()
        # plt.title('diff_blured')
        # plt.imshow(np.abs(r_blured - i_blured), cmap='gray')

        # alignment
        from alignment.Aligner import Aligner
        aligner = Aligner()
        matches_image, warped, tform, _ = aligner.align_using_feature_matching(static=r_blured, moving=i_blured)
        print(f"tform: {tform}")

        aligner.align_using_normxcorr(static=r_blured, moving=i_blured)


        # plt.figure()
        # plt.title('matches_image')
        # plt.imshow(matches_image, cmap='gray')
        # plt.figure()
        # plt.title('warped')
        # plt.imshow(warped, cmap='gray')

        # warped_diff = np.abs(warped - reference)
        # plt.figure()
        # plt.title('warped_diff')
        # plt.imshow(warped_diff, cmap='gray')

        from segmentation.Segmenter import Segmenter
        segmenter = Segmenter()
        threshold_segmentation, hist, smooth_hist, low_thres, high_thres = segmenter.segment_image_by_threshold(
            reference)
        threshold_segmentation = noise_cleaner.majority(threshold_segmentation, radius=2)

        if hist is not None:
            plt.figure()
            plt.title(f"chosen thresholds: {low_thres}, {high_thres}")
            plt.plot(hist, color="blue")
            plt.plot(smooth_hist, color="red")
            plt.show()

        plt.figure()
        plt.title("threshold_segmentation")
        plt.imshow(threshold_segmentation)


        plt.show()
    main()
