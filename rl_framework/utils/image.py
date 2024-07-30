import cv2
from numpy import ndarray

from .pipeline import Operation


class ImageResize(Operation):
    def __init__(self, height: int, width: int, *args, **kwargs):
        super().__init__()
        self.height = height
        self.width = width
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: ndarray) -> ndarray:
        x = cv2.resize(x, (self.height, self.width), *self.args, **self.kwargs)
        return x


class ImageToGray(Operation):
    def forward(self, x: ndarray) -> ndarray:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return x


class ImageNormalize(Operation):
    def forward(self, x: ndarray) -> ndarray:
        x = x / 255.0
        return x
