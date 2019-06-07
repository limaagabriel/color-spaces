import torch
from torch import nn
from skimage import color
from abc import ABC, abstractmethod


class ColorSpace(ABC):
	@abstractmethod
	def from_rgb(self, image):
		pass

	@staticmethod
	def build(name):
		subclasses = ColorSpace.__subclasses__()
		if name not in map(lambda x: x.__name__, subclasses):
			raise ValueError('Color space named {} not found'.format(name))
		return list(filter(lambda x: x.__name__ == name, subclasses))[0]


class RGB(ColorSpace):
	def from_rgb(self, image):
		return image.copy()


class CIELab(ColorSpace):
	def from_rgb(self, image):
		return color.rgb2lab(image)


class HSV(ColorSpace):
	def from_rgb(self, image):
		return color.rgb2hsv(image)


class ColorNet(nn.Module):
	def __init__(self, color_spaces):
		super(ColorNet, self).__init__()

	def forward(self, x):
		return x