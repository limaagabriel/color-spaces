import os
import torch
from torch import nn, optim
from model.densenet import DenseNet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

growth = 32
batch_size = 32
num_classes = 3
num_input_features = 3

epochs = 100000

train_dataset = ImageNet(os.environ.get('IMAGENET_PATH'),
							split='train', download=True)
valid_dataset = ImageNet(os.environ.get('IMAGENET_PATH'),
							split='val', download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


model = DenseNet(num_input_features, growth, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(epochs):
	train_loss = 0
	valid_loss = 0

	model.train()
	for x, y in train_loader:
		optimizer.zero_grad()

		yhat = model(x)
		loss = criterion(yhat, y)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	train_loss = train_loss / len(train_loader)

	model.eval()
	with torch.no_grad():
		for x, y in valid_loader:
			yhat = model(x)
			loss = criterion(yhat, y)
			valid_loss += loss.item()
		valid_loss = valid_loss / len(valid_loader)



