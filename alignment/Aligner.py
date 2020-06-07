import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np
from scipy.signal import fftconvolve
from noise_cleaning.NoiseCleaner import NoiseCleaner


class Aligner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._is_force_translation = self._config.alignment.is_force_translation
        self._subpixel_accuracy_resolution = self._config.alignment.subpixel_accuracy_resolution

        self._detector = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._noise_cleaner = NoiseCleaner()

    def align_using_feature_matching(self, static, moving):
        kp1, des1 = self._detector.detectAndCompute(static, None)
        kp2, des2 = self._detector.detectAndCompute(moving, None)
        matches = self._matcher.match(des1, des2)

        assert len(matches) >= 4  # for perspective tform
        moving_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        static_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        tform, mask = cv2.findHomography(moving_pts, static_pts, cv2.RANSAC, 5.0)
        tform = self._force_translation_only(tform)
        matches_mask = mask.ravel().tolist()

        matches_image = cv2.drawMatches(static, kp1, moving, kp2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                        matchesMask=matches_mask,
                                        matchColor=(0, 255, 0), )

        itform = np.linalg.pinv(tform)
        # return matches_image
        return matches_image, itform

    def align_using_tform(self, static, moving, tform):
        warped = cv2.warpPerspective(moving, tform, (static.shape[1], static.shape[0]))
        warped_region_mask = self._find_warped_region(tform, moving, static)
        return warped, warped_region_mask

    def align_using_shift(self, static, moving, shift_xy):
        tform = np.eye(3)
        tform[0, 2] = shift_xy[1]
        tform[1, 2] = shift_xy[0]
        warped, warped_region_mask = self.align_using_tform(static, moving, tform)
        return warped, warped_region_mask

    def _find_warped_region(self, tform, moving, static):
        white = np.ones(moving.shape)
        warped = cv2.warpPerspective(white, tform, (static.shape[1], static.shape[0]))
        warped_region_mask = warped > 0
        return warped_region_mask

    def _force_translation_only(self, tform):
        if self._is_force_translation:
            tform[0, 1] = 0
            tform[1, 0] = 0
            tform[2, 0] = 0
            tform[2, 1] = 0

            tform[0, 0] = 1
            tform[1, 1] = 1
            print(f"forcing tform {tform} to translation only")

        return tform

    import numpy as np
    from scipy.signal import fftconvolve

    @staticmethod
    def normxcorr2(template, image, mode="full"):
        """
        credit: https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
        Input arrays should be floating point numbers.
        :param template: N-D array, of template or filter you are using for cross-correlation.
        Must be less or equal dimensions to image.
        Length of each dimension must be less than length of image.
        :param image: N-D array
        :param mode: Options, "full", "valid", "same"
        full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
        Output size will be image size + 1/2 template size in each dimension.
        valid: The output consists only of those elements that do not rely on the zero-padding.
        same: The output is the same size as image, centered with respect to the ‘full’ output.
        :return: N-D array of same dimensions as image. Size depends on mode parameter.
        """

        # If this happens, it is probably a mistake
        if np.ndim(template) > np.ndim(image) or \
                len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
            print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

        template = template - np.mean(template)
        image = image - np.mean(image)

        a1 = np.ones(template.shape)
        # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
        ar = np.flipud(np.fliplr(template))
        out = fftconvolve(image, ar.conj(), mode=mode)

        image = fftconvolve(np.square(image), a1, mode=mode) - \
                np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

        # Remove small machine precision errors after subtraction
        image[np.where(image < 0)] = 0

        template = np.sum(np.square(template))
        out = out / np.sqrt(image * template)

        # Remove any divisions by 0 or very close to 0
        out[np.where(np.logical_not(np.isfinite(out)))] = 0

        return out

    def align_using_normxcorr(self, static, moving):
        """
        normxcorr is same as
        # conv_res = convolve2d(static, moving, mode='full')
        # best_location_conv = np.unravel_index(np.argmax(res), res.shape)
        but fast.
        """
        res = self.normxcorr2(static, moving, mode='full')
        best_location = np.unravel_index(np.argmax(res), res.shape)
        moving_should_be_strided_by = -(np.array(best_location) - np.array(moving.shape) - 1)

        return moving_should_be_strided_by

    def align_using_ecc(self, static, moving):
        number_of_iterations = 100
        termination_eps = 1e-10

        cc, tform = cv2.findTransformECC(
            static,
            moving,
            np.eye(3, dtype=np.float32)[:2, :],
            cv2.MOTION_TRANSLATION,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps),
            inputMask=np.ones(static.shape, dtype='uint8') * 255,
            gaussFiltSize=11
        )
        return tform

    def align_images(self, static, moving):
        # clean noise
        static_clean = self._noise_cleaner.clean_salt_and_pepper(static, 5)
        moving_clean = self._noise_cleaner.clean_salt_and_pepper(moving, 5)
        static_eq = self._noise_cleaner.equalize_histogram(static_clean.astype('uint8'))
        moving_eq = self._noise_cleaner.equalize_histogram(moving_clean.astype('uint8'))

        # enlarge to obtain subpixel accuracy
        static_enlarged = cv2.resize(static_eq,
                                     (0, 0),
                                     fx=self._subpixel_accuracy_resolution,
                                     fy=self._subpixel_accuracy_resolution)
        moving_enlarged = cv2.resize(moving_eq,
                                    (0, 0),
                                    fx=self._subpixel_accuracy_resolution,
                                    fy=self._subpixel_accuracy_resolution)

        # normxcorr alignment (translation only)
        moving_should_be_strided_by_10 = self.align_using_normxcorr(static=static_enlarged,
                                                                    moving=moving_enlarged)

        # return to normal size of translation
        moving_should_be_strided_by = np.array(moving_should_be_strided_by_10) / self._subpixel_accuracy_resolution

        # perform actual warp
        warped, warp_mask = self.align_using_shift(static, moving, moving_should_be_strided_by)

        return warped, warp_mask
