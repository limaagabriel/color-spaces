import os
import sys
import torch
import shutil
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from data.patch import PatchDataset
from model.densenet import DenseNet
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

message = 'Running on {} device (cuda.is_available={})'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(message.format(device, torch.cuda.is_available()))

# Model parameters
train = False
growth = 32
batch_size = 4
learning_rate = 1e-4
num_input_features = 3

# Training parameters
epochs = 100000

# Input preprocessing
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor()
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

best_validation_loss = sys.maxsize
criterion = nn.CrossEntropyLoss()
model_path =  os.path.join('model', 'saved_models')
model = DenseNet(num_input_features, growth, train_dataset.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if not os.path.exists(model_path):
	os.mkdir(model_path)
else:
	shutil.rmtree(model_path)

for epoch in range(epochs):
	if not train:
		break
	train_loss = 0
	valid_loss = 0

	model.train()
	for x, y in train_loader:
		x = x.to(device)
		y = y.to(device)
		optimizer.zero_grad()

		yhat = model(x).to(device)
		loss = criterion(yhat, y)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
	train_loss = train_loss / len(train_loader)

	model.eval()
	with torch.no_grad():
		for x, y in valid_loader:
			x = x.to(device)
			y = y.to(device)

			yhat = model(x).to(device)
			loss = criterion(yhat, y)
			valid_loss += loss.item()
		valid_loss = valid_loss / len(valid_loader)

		if valid_loss < best_validation_loss:
			best_validation_loss = valid_loss
			path = os.path.join(model_path, '{}.pth'.format(model.__class__.__name__))

			if os.path.exists(path):
				os.remove(path)

			torch.save(model.state_dict(), path)

	print('Training loss: {}\t Validation loss: {}'.format(train_loss, valid_loss))

model.eval()
with torch.no_grad():
	test_acc = 0

	for x, y in test_loader:
		if train:
			break

		x = x.to(device)
		y = y.to(device)
		yhat = F.softmax(model(x).to(device), dim=1).argmax(dim=1)
		test_acc += torch.sum(yhat == y)
	
	test_acc = test_acc / len(test_loader)
	print(test_acc)
		

