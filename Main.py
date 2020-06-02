from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    def main():
        print("hola")
        config = ConfigProvider.config()
        assert os.path.isdir(config.data.defective_examples_folder_path)
        assert os.path.isdir(config.data.non_defective_examples_folder_path)

        inspected = cv2.imread(config.data.inspected_image_path, 0)
        reference = cv2.imread(config.data.reference_image_path, 0)
        diff = np.abs(inspected - reference)
        #
        # cv2.imshow("inspected", inspected)
        # cv2.imshow("reference", reference)
        # cv2.imshow("diff", diff)

        plt.figure()
        plt.title('inspected')
        plt.imshow(inspected)
        plt.show()

        cv2.waitKey(0)
    main()
