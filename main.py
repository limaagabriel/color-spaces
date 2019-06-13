import os
import sys
import torch
import shutil

import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from data.datasets.h5 import H5Dataset
from data.preprocessing import transforms

from model.stop import StopCriterion
from model.densenet import DenseNetClassifier

# Training parameters
epochs = 100000
batch_size = 8

# Input preprocessing
root = os.environ.get('OBJECT_DETECTION_DATASET_PATH')

model_parameters = {
	'growth': 32,
	'bottleneck': 4,
	'in_features': 3,
	'num_classes': 7,
	'compression': 0.5
}

optimizer_parameters = {
	'lr': 1e-4
}

for color_space in ['CIELab', 'HSV', 'RGB']:
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToPILImage(),
		torchvision.transforms.Resize((224, 224)),
		transforms.ToColorSpace(color_space),
		torchvision.transforms.ToTensor()
	])

	train_dataset = H5Dataset(root, split='train', transform=transform)
	valid_dataset = H5Dataset(root, split='valid', transform=transform)
	test_dataset = H5Dataset(root, split='test', transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

	stop_criterion = StopCriterion.valid_loss(0.01, max_iterations=epochs)
	model = DenseNetClassifier(**model_parameters, verbose=True)
	model.set_optimizer(optim.Adam, **optimizer_parameters)
	model.set_loss_criterion(nn.CrossEntropyLoss)

	model.fit(train_loader, valid_loader, stop_criterion)
	score = model.score_from_loader(test_loader)
	model.save('densenet_{}_{}.pth'.format(color_space, score))
	print('Color space \'{}\': {}'.format(color_space, score))

