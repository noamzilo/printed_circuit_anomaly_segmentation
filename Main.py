from Utils.ConfigProvider import ConfigProvider
import cv2
from matplotlib import pyplot as plt
import matplotlib
from Utils.plotting.plot_utils import show_color_diff
from Utils.plotting.plot_utils import plot_image
from alignment.Aligner import Aligner
from segmentation.Segmenter import Segmenter


from defect_segmentation.DefectSegmenter import DefectSegmenter


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


if __name__ == "__main__":
    def main():
        config = ConfigProvider.config()

        # read data
        inspected = cv2.imread(config.data.defective_inspected_path1, 0).astype('float32')
        reference = cv2.imread(config.data.defective_reference_path1, 0).astype('float32')
        # inspected = cv2.imread(config.data.defective_inspected_path2, 0).astype('float32')
        # reference = cv2.imread(config.data.defective_reference_path2, 0).astype('float32')
        # inspected = cv2.imread(config.data.non_defective_inspected_path, 0).astype('float32')
        # reference = cv2.imread(config.data.non_defective_reference_path, 0).astype('float32')

        # registration
        aligner = Aligner()
        warped, warp_mask = aligner.align_images(static=inspected, moving=reference)

        defect_segmenter = DefectSegmenter()
        defect_mask = defect_segmenter.segment_defects(inspected, warped, warp_mask)

        cv2.imshow("inspected", inspected)
        cv2.imshow("reference", reference)
        cv2.imshow("result", defect_mask)
        cv2.waitKey(0)

        plt.show()
    main()
