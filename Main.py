from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    def main():
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

        inspected = cv2.imread(config.data.inspected_image_path, 0)
        reference = cv2.imread(config.data.reference_image_path, 0)
        diff = np.abs(inspected - reference)

        plt.figure()
        plt.title('inspected')
        plt.imshow(inspected)
        plt.figure()
        plt.title('reference')
        plt.imshow(reference)
        plt.figure()
        plt.title('diff')
        plt.imshow(diff)


        r_blured = cv2.blur(reference, (config.alignment.blur_radius, config.alignment.blur_radius))
        i_blured = cv2.blur(inspected, (config.alignment.blur_radius, config.alignment.blur_radius))

        from alignment.Aligner import Aligner
        aligner = Aligner()
        matches_image, warped = aligner.align_using_feature_matching(static=r_blured, moving=i_blured)
        plt.figure()
        plt.title('matches_image')
        plt.imshow(matches_image)
        plt.figure()
        plt.title('warped')
        plt.imshow(warped)

        warped_diff = np.abs(warped - reference)
        plt.figure()
        plt.title('warped_diff')
        plt.imshow(warped_diff)

        plt.show()
    main()
