from abc import ABC, abstractmethod


class BaseSegmenter(ABC):
    def init(self):
        pass

    @abstractmethod
    def detect(self, inspected, warped, warp_mask):
        raise NotImplementedError
