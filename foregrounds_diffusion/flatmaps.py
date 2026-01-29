import numpy as np
import sys
import os
from scipy import ndimage

class FlatSkyMap:
    def __init__(self, height: int, width: int, pixel_size: float):
        self.height = height
        self.width = width
        self.pixel_size = pixel_size  # in arcminutes
        self.data = np.zeros((height, width))

    def get_height(self) -> int:
        return self.height

    def get_width(self) -> int:
        return self.width

    def get_pixel_size(self) -> float:
        return self.pixel_size

    def get_num_pixels(self) -> int:
        return self.height * self.width

class CIBMap(FlatSkyMap):
    def __init__(self, height: int, width: int, pixel_size: float, frequency: float):
        super().__init__(height, width, pixel_size)
        self.frequency = frequency  # in GHz

    def get_frequency(self) -> float:
        return self.frequency

class TSZMap(FlatSkyMap):
    def __init__(self, height: int, width: int, pixel_size: float, frequency: float):
        super().__init__(height, width, pixel_size)
        self.frequency = frequency  # in GHz

    def get_frequency(self) -> float:
        return self.frequency

class SimMap(FlatSkyMap):
    def __init__(self):
        super().__init__(height=256, width=256, pixel_size=1.0) # Default values


class DiffusionMap(FlatSkyMap):
    def __init__(self):
        super().__init__(height=256, width=256, pixel_size=1.0) # Default values