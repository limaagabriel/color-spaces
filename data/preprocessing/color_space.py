from skimage import color
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
		return (color.rgb2lab(image) * 255).astype('uint8')


class HSV(ColorSpace):
	def from_rgb(self, image):
		return (color.rgb2hsv(image) * 255).astype('uint8')
