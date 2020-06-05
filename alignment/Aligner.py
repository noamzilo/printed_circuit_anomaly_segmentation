import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np


class Aligner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._blur_radius = self._config.alignment.blur_radius
        self._min_match_distance = self._config.alignment.min_match_distance
        self._is_force_translation = self._config.alignment.is_force_translation

        self._detector = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
                                        matchColor=(0, 255, 0),)

        itform = np.linalg.pinv(tform)
        warped = cv2.warpPerspective(moving, itform, (static.shape[1], static.shape[0]))

        warped_region_mask = self._find_warped_region(itform, moving, static)
        # return matches_image
        return matches_image, warped, tform, warped_region_mask

    # def _find_warped_region(self, tform, moving):
        # h, w = moving.shape
        # moving_corners = np.array([
        #     [0, 0,   h-1, h-1],
        #     [0, w-1, w-1, 0],
        #     [1, 1,   1,   1],
        # ])
        # warped_corners = np.matmul(tform, moving_corners)
        # warped_corners_locations = np.vstack([warped_corners[0, :] / warped_corners[2, :],
        #                                      warped_corners[1, :] / warped_corners[2, :],])
        # warped_corners_locations = np.array(warped_corners_locations, dtype=np.int)
        # # warped_corners_locations[warped_corners_locations < 0] = 0
        #
        # return warped_corners_locations

    def _find_warped_region(self, tform, moving, static):
        # this is very ugly, but I wanted to get on with it and not waste any more time.
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

    def align_using_normxcorr(self, static, moving):
        raise NotImplementedError
