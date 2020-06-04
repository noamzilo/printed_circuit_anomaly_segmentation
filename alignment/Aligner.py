import cv2
from Utils.ConfigProvider import ConfigProvider
import numpy as np


class Aligner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._blur_radius = self._config.alignment.blur_radius
        self._min_match_distance = self._config.alignment.min_match_distance

        self._detector = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_using_feature_matching(self, static, moving):
        kp1, des1 = self._detector.detectAndCompute(static, None)
        kp2, des2 = self._detector.detectAndCompute(moving, None)
        matches = self._matcher.match(des1, des2)

        # matches = sorted(filter(lambda x: x.distance < self._min_match_distance, matches), key=lambda x: x.distance)
        # matches_image = cv2.drawMatches(static, kp1, moving, kp2, matches[:10], None,
        #                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        assert len(matches) >= 3  # for affine tform
        moving_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        static_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        tform, mask = cv2.findHomography(moving_pts, static_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        matches_image = cv2.drawMatches(static, kp1, moving, kp2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                        matchesMask=matches_mask,
                                        matchColor=(0, 255, 0),)

        warped = cv2.warpPerspective(moving, np.linalg.pinv(tform), (static.shape[1], static.shape[0]))

        # return matches_image
        return matches_image, warped

    def align_using_normxcorr(self, static, moving):
        pass
