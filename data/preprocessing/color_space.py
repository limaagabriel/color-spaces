import cv2
from abc import ABC, abstractmethod


class ColorSpace(ABC):
	@abstractmethod
	def from_rgb(self, image):
		pass


class ColorSpaceFactory(object):
	@staticmethod
	def build(name):
		subclasses = ColorSpace.__subclasses__()
		if name not in map(lambda x: x.__name__, subclasses):
			raise ValueError('Color space named {} not found'.format(name))
		return list(filter(lambda x: x.__name__ == name, subclasses))[0]()


class RGB(ColorSpace):
	def from_rgb(self, image):
		return image.copy()


class CIELab(ColorSpace):
	def from_rgb(self, image):
		return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


class HSV(ColorSpace):
	def from_rgb(self, image):
		return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
