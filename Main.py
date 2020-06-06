from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy.misc


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

        inspected = noise_cleaner.clean_salt_and_pepper(inspected, 5)
        reference = noise_cleaner.clean_salt_and_pepper(reference, 5)



        # alignment
        from alignment.Aligner import Aligner
        aligner = Aligner()
        # tform = aligner.align_using_ecc(inspected, reference)
        # tform = np.hstack([tform, np.array([0, 0, 1])])
        # print(f"tform: {tform}")

        # aligner.align_using_tform(reference, inspected, tform)

        resize = 5
        moving_should_be_strided_by_10 = aligner.align_using_normxcorr(cv2.resize(reference,
                                                                               (0, 0),
                                                                               fx=resize,
                                                                               fy=resize),
                                                                    cv2.resize(inspected, (0, 0), fx=resize, fy=resize))
        moving_should_be_strided_by = np.array(moving_should_be_strided_by_10) / resize


        warped, warp_mask = aligner.align_using_shift(reference, inspected, moving_should_be_strided_by)
        plt.figure()
        plt.title("warped")
        plt.imshow(warped)
        # plt.show()

        diff = np.abs((np.float32(warped) - np.float32(reference)))

        # xx, yy = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(xx, yy, diff, rstride=1, cstride=1, cmap=plt.cm.gray,
        #                 linewidth=0)
        # plt.show()


        plt.figure()
        plt.title("diff")
        plt.imshow(diff)
        # plt.show()

        to_show = np.ones((warped.shape[0], warped.shape[1], 3))
        to_show[:, :, 0] = warped
        to_show[:, :, 1] = reference
        to_show[:, :, 2] = warped
        to_show = to_show.astype('uint8')

        plt.figure()
        plt.title("to_show")
        plt.imshow(to_show)
        # plt.show()

        diff_blured = noise_cleaner.blur(diff, sigma=5)
        plt.figure()
        plt.title("diff_blured")
        plt.imshow(diff_blured)
        # plt.show()

        edges = cv2.Canny(reference.astype('uint8'), 100, 200) > 0
        edges_dialated = noise_cleaner.dilate(edges.astype(np.float32), 3)
        diff_no_edges = diff.copy()
        diff_no_edges[edges_dialated > 0] = 0

        plt.figure()
        plt.title("diff_no_edges")
        plt.imshow(diff_no_edges)
        # plt.show()
        #

        result_mask = diff_no_edges > 25
        plt.figure()
        plt.title("result_mask")
        plt.imshow(result_mask)
        # diff_show = diff_no_edges[100:300, 50:400]
        # xx, yy = np.mgrid[0:diff_show.shape[0], 0:diff_show.shape[1]]
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(xx, yy, diff_show, rstride=1, cstride=1, cmap=plt.cm.gray,
        #                 linewidth=0)
        plt.show()

        hi=5
    main()
