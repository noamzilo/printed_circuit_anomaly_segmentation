import cv2
from Utils.ConfigProvider import ConfigProvider


class Aligner(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._blur_radius = self._config.alignment.blur_radius

        self._detector = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, static, moving):
        s_blured = cv2.blur(static, (self._blur_radius, self._blur_radius))
        m_blured = cv2.blur(moving, (self._blur_radius, self._blur_radius))

        kp1, des1 = self._detector.detectAndCompute(s_blured, None)
        kp2, des2 = self._detector.detectAndCompute(m_blured, None)
        matches = self._matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches_image = cv2.drawMatches(static, kp1, moving, kp2, matches[:10], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return matches_image

