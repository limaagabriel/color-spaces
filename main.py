import os
import torch
import shutil
from torch import nn, optim
from torchvision import transforms
from data.patch import PatchDataset
from model.densenet import DenseNet
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on {} device'.format(device))

# Model parameters
growth = 32
batch_size = 4
learning_rate = 0.01
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

best_validation_loss = 0
criterion = nn.CrossEntropyLoss()
model_path =  os.path.join('model', 'saved_models')
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
model = DenseNet(num_input_features, growth, train_dataset.num_classes).to(device)

if not os.path.exists(model_path):
	os.mkdir(model_path)
else:
	shutil.rmtree(model_path)

for epoch in range(epochs):
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
			path = os.path.join(model_path, '{}.pth'.format(model.__class__.__name__))
			torch.save(model.state_dict(), path)

	print('Training loss: {}\t Validation loss: {}'.format(train_loss, valid_loss))


