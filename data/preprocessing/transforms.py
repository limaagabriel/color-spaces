import numpy as np
from PIL import Image
from data.preprocessing.color_space import ColorSpaceFactory


class ToColorSpace(object):
	def __init__(self, color_space_id):
		self.color_space = ColorSpaceFactory.build(color_space_id)

	def __call__(self, sample):
		mat = np.asarray(sample)
		return self.color_space.from_rgb(mat)
