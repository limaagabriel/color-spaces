import os
import sys
import torch
import shutil

import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from data.datasets.patch import PatchDataset
from model.densenet import DenseNetClassifier
from data.preprocessing import transforms

# Training parameters
epochs = 1
batch_size = 8

# Input preprocessing
transform = torchvision.transforms.Compose([
	torchvision.transforms.Resize((224, 224)),
	torchvision.transforms.ToTensor()
])

train_dataset = PatchDataset(os.environ.get('OBJECT_DETECTION_DATASET_PATH'),
								split='train', transform=transform)
valid_dataset = PatchDataset(os.environ.get('OBJECT_DETECTION_DATASET_PATH'),
								split='valid', transform=transform)
test_dataset = PatchDataset(os.environ.get('OBJECT_DETECTION_DATASET_PATH'),
								split='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_parameters = {
	'growth': 32,
	'bottleneck': 4,
	'in_features': 3,
	'compression': 0.5,
	'num_classes': train_dataset.num_classes,
}

optimizer_parameters = {
	'lr': 1e-4,
	'weight_decay': 0.001
}

model_path =  os.path.join('model', 'saved_models')

model = DenseNetClassifier(**model_parameters, verbose=True)
model.set_optimizer(optim.Adam, **optimizer_parameters)
model.set_loss_criterion(nn.CrossEntropyLoss)

model.fit(epochs, train_loader, valid_loader)
print(model.score_loader(test_loader))

