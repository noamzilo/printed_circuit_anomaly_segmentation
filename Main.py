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

        # plt.figure()
        # plt.title('inspected')
        # plt.imshow(inspected)
        # plt.figure()
        # plt.title('reference')
        # plt.imshow(reference)
        # plt.figure()
        # plt.title('diff')
        # plt.imshow(diff)
        #
        # plt.show()

        from alignment.Aligner import Aligner
        aligner = Aligner()
        matches = aligner.align(reference, inspected)
        plt.figure()
        plt.title('matches')
        plt.imshow(matches)

        plt.show()
    main()
