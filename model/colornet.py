import torch
from torch import nn


class ColorNet(nn.Module):
	def __init__(self, color_spaces):
		super(ColorNet, self).__init__()

	def forward(self, x):
		return x