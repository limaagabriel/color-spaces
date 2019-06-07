import torch
from torch import nn
from functools import reduce
import torch.nn.functional as F

class DenseLayer(nn.Module):
	def __init__(self, in_features, depth, growth, bottleneck, compression):
		super(DenseLayer, self).__init__()

		self.depth = depth
		self.growth = growth
		self.in_features = in_features
		self.bottleneck_factor = bottleneck
		self.compression_factor = compression
		self.use_bottleneck_layer = bottleneck is not None
		self.layers = nn.Sequential(*list(map(self.__mapper, range(depth))))

	def output(self):
		return self.in_features + (self.depth * self.growth)

	def __mapper(self, l):
		in_features = self.in_features + (l * self.growth)
		out_features = self.growth
		layers = []

		if self.use_bottleneck_layer:
			inter_features = out_features * self.bottleneck_factor

			layers = [
				nn.BatchNorm2d(in_features),
				nn.ReLU(),
				nn.Conv2d(in_features, inter_features, 1)
			]

			in_features = inter_features

		layers = layers + [
			nn.BatchNorm2d(in_features),
			nn.ReLU(),
			nn.Conv2d(in_features, out_features, 3, padding=1)

		]

		return nn.Sequential(*layers)
		

	def forward(self, x):
		def reducer(current_x, layer):
			t = layer(current_x)

			if self.compression_factor > 0:
				t = F.dropout(t, p=self.compression_factor,
								training=self.training)
			return torch.cat([current_x, t])

		return reduce(reducer, self.layers, x)

class DenseNet(nn.Module):
	def __init__(self, in_features, growth, num_classes,
					block_config=(6, 12, 24, 16),
					bottleneck=4, compression=0.5):
		super(DenseNet, self).__init__()

		self.growth = growth
		self.bottleneck = bottleneck
		self.compression = compression
		self.block_config = block_config

		before_dense_blocks, out_f = self.__get_before_dense_blocks(in_features)
		dense_blocks, out_f = self.__get_dense_blocks(out_f)
		after_dense_blocks, out_f = self.__get_after_dense_blocks(out_f)

		layers = before_dense_blocks + dense_blocks + after_dense_blocks
		self.layers = nn.Sequential(*layers)
		self.classifier = nn.Linear(out_f, num_classes)

	def __get_transition_layer(self, features):
		return nn.Sequential(
			nn.Conv2d(features, features, 1),
			nn.AvgPool2d(2)
		)

	def __get_before_dense_blocks(self, in_features):
		out_features = 2 * self.growth

		layers = [
			nn.Conv2d(in_features, out_features, 7, stride=2, padding=3),
			nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		]

		return layers, out_features

	def __get_dense_blocks(self, in_features):
		def mapper(pack):
			index, depth = pack
			acc_config = sum(self.block_config[:index])
			in_f = in_features + (acc_config * self.growth)
			dense_layer = DenseLayer(in_f, depth, self.growth,
					bottleneck=self.bottleneck, compression=self.compression)

			if index >= len(self.block_config) - 1:
				return dense_layer

			return nn.Sequential(
				dense_layer,
				self.__get_transition_layer(dense_layer.output())
			)

		layers = list(map(mapper, enumerate(self.block_config)))
		return layers, (in_features + (sum(self.block_config) * self.growth))

	def __get_after_dense_blocks(self, in_features):
		return [ nn.BatchNorm2d(in_features) ], in_features

	def forward(self, x):
		y = self.layers(x)
		return self.classifier(y)

