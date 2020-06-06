from Utils.ConfigProvider import ConfigProvider
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
config = ConfigProvider.config()

inspected1 = cv2.imread(config.data.non_defective_inspected_path, 0)
reference1 = cv2.imread(config.data.non_defective_reference_path, 0)
inspected2 = cv2.imread(config.data.defective_inspected_path1, 0)
reference2 = cv2.imread(config.data.defective_reference_path1, 0)
inspected3 = cv2.imread(config.data.defective_inspected_path2, 0)
reference3 = cv2.imread(config.data.defective_reference_path2, 0)

inspected = inspected2
reference = reference2

diff = np.abs(np.array(inspected, dtype=np.int) - np.array(reference, dtype=np.int))

plt.close('all')

plt.figure()
plt.title('inspected')
plt.imshow(inspected, cmap='gray')
plt.figure()
plt.title('reference')
plt.imshow(reference, cmap='gray')
plt.figure()
plt.title('diff')
plt.imshow(diff, cmap='gray')

plt.show()

from noise_cleaning.NoiseCleaner import NoiseCleaner
noise_cleaner = NoiseCleaner()
r_clean = reference
i_clean = inspected

r_clean = noise_cleaner.clean_salt_and_pepper(r_clean, radius=11)
i_clean = noise_cleaner.clean_salt_and_pepper(i_clean, radius=11)


plt.figure()
plt.title("cleaning did:")
plt.imshow((i_clean.astype("int") - inspected.astype("int")).astype("uint8"))
plt.show()

plt.figure()
plt.title('i_clean')
plt.imshow(i_clean, cmap='gray')


from alignment.Aligner import Aligner
aligner = Aligner()
matches_image, tform = aligner.align_using_feature_matching(static=r_clean, moving=i_clean)
clean_warped, warped_mask = aligner.align_using_tform(r_clean, i_clean, tform)
print(f"tform: {tform}")
print(f"moving should be strided {tform[0, 2]} down and {tform[1, 2]} right ")

plt.figure()
plt.title('matches_image')
plt.imshow(matches_image, cmap='gray')
plt.figure()
plt.title('clean_warped')
plt.imshow(clean_warped, cmap='gray')

warped_diff = np.ones(inspected.shape) * 255
warped_diff[warped_mask] = np.array(clean_warped, dtype=np.int)[warped_mask]
warped_diff[warped_mask] -= np.array(r_clean, dtype=np.int)[warped_mask]
warped_diff = np.array(np.abs(warped_diff), dtype=np.uint8)

plt.figure()
plt.title('warped_diff')
plt.imshow(warped_diff, cmap='gray')
plt.show()


from alignment.Aligner import Aligner
aligner = Aligner()
moving_should_be_strided_by = aligner.align_using_normxcorr(r_clean, i_clean)
print(f"moving_should_be_strided_by: {moving_should_be_strided_by}")
clean_warped_xcorr, warped_mask_xcorr = aligner.align_using_shift(r_clean, i_clean, moving_should_be_strided_by)
warped_diff_xcorr = np.ones(inspected.shape) * 255
warped_diff_xcorr[warped_mask_xcorr] = np.array(clean_warped_xcorr, dtype=np.int)[warped_mask_xcorr]
warped_diff_xcorr[warped_mask_xcorr] -= np.array(r_clean, dtype=np.int)[warped_mask_xcorr]
warped_diff_xcorr = np.array(np.abs(warped_diff_xcorr), dtype=np.uint8)

plt.figure()
plt.title('warped_diff_xcorr')
plt.imshow(warped_diff_xcorr.astype('uint8'), cmap='gray')
plt.show()


from segmentation.Segmenter import Segmenter
segmenter = Segmenter()
kmeans_segmentation = segmenter.segment_image_by_kmeans(reference)
# kmeans_segmentation = noise_cleaner.bilateral_filter(kmeans_segmentation, sigma_spatial=15)
# uniq = np.unique(kmeans_segmentation)
# for i, val in enumerate(uniq):
#     kmeans_segmentation[kmeans_segmentation == val] = i
kmeans_segmentation = noise_cleaner.majority(kmeans_segmentation.astype('uint8'), radius=3)

plt.figure()
plt.title("kmeans_segmentation")
plt.imshow(kmeans_segmentation)
plt.show()

segmentation_map = kmeans_segmentation

warped, warped_region_mask = aligner.align_using_tform(static=reference, moving=inspected, tform=tform)
# warped, warped_region_mask = aligner.align_using_shift(static=reference, moving=inspected, shift_xy=moving_should_be_strided_by)

dan_diff = np.abs(reference.astype('float32') / 255 - warped.astype('float32') / 255)
dan_diff[~warped_region_mask] = 0

plt.figure()
plt.title("dan_diff")
plt.imshow(dan_diff, cmap='gray')
plt.show()



#
#
# warped_diff1 = np.ones(inspected.shape) * 255
# warped_diff1[warped_region_mask] = np.array(warped, dtype=np.int)[warped_region_mask]
# warped_diff1[warped_region_mask] -= np.array(reference, dtype=np.int)[warped_region_mask]
# warped_diff1 = np.array(np.abs(warped_diff1), dtype=np.uint8)
#
# plt.figure()
# plt.title("warped")
# plt.imshow(warped, cmap='gray')
# plt.show()
#
# original_warped_diff = warped_diff1.copy()
# original_warped_diff[~warped_region_mask] = 0
# plt.figure()
# plt.imshow(original_warped_diff, cmap='gray')
# plt.show()
#
# for c in range(config.segmentation.num_classes):
#     temp_diff = original_warped_diff.copy()
#     temp_diff[~(segmentation_map==c)] = 0
#
#     class_c_thres = 30
#     plt.figure()
#     plt.title(f'class {c} defects mask')
#     defects_class_c = np.zeros_like(temp_diff)
#     defects_class_c[class_c_thres < temp_diff] = 255
#     defects_class_c[temp_diff <= class_c_thres] = 0
#     plt.imshow(defects_class_c)
#
#
# plt.show()
#
#
