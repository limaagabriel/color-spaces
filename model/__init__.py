import sys
import torch
import torch.nn.functional as F

from abc import ABC


class Classifier(ABC):
	module = None

	def __init__(self, verbose=False, *args, **kwargs):
		device_id = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.__params = { 'args': args, 'kwargs': kwargs }
		self.__device = torch.device(device_id)

		self.__model = self.module(*args, **kwargs).to(self.__device)
		self.__best_model = None
		self.__optimizer = None
		self.__criterion = None
		self.__verbose = verbose

		if self.__verbose:
			message = 'Using {} device (cuda.is_available={})'
			print(message.format(self.__device, torch.cuda.is_available()))

	@property
	def trained_model(self):
		return self.__best_model

	def set_optimizer(self, optimizer, **kwargs):
		self.__optimizer = optimizer(self.__model.parameters(), **kwargs)

	def set_loss_criterion(self, criterion, **kwargs):
		self.__criterion = criterion(**kwargs)

	def __verify_training_dependencies(self):
		if self.__optimizer is None:
			raise ValueError('The optimizer must be defined before this operation')
		if self.__criterion is None:
			raise ValueError('The loss criterion must be defined before this operation')
	
	def __verify_predict_dependencies(self):
		if self.__best_model is None:
			raise ValueError('The classifier must be trained before predicting')

	def __train_step(self, loader):
		train_loss = 0
		self.__model.train()
		for x, y in loader:
			x = x.to(self.__device)
			y = y.to(self.__device)
			self.__optimizer.zero_grad()

			yhat = self.__model(x)
			loss = self.__criterion(yhat, y)
			loss.backward()
			self.__optimizer.step()

			train_loss += loss.item()
		train_loss = train_loss / len(loader)
		return train_loss

	def __valid_step(self, loader):
		valid_loss = 0
		self.__model.eval()
		with torch.no_grad():
			for x, y in loader:
				x = x.to(self.__device)
				y = y.to(self.__device)
					
				yhat = self.__model(x)
				loss = self.__criterion(yhat, y)
				valid_loss += loss.item()
			valid_loss = valid_loss / len(loader)
		return valid_loss

	def __save_model(self, current_loss, best_loss):
		if best_loss < current_loss:
			args = self.__params['args']
			kwargs = self.__params['kwargs']
			self.__best_model = self.module(*args, **kwargs)
			self.__best_model.load_state_dict(self.__model.state_dict())

			return best_loss
		return current_loss

	def fit(self, epochs, train_loader, valid_loader):
		best_valid_loss = sys.maxsize
		self.__verify_training_dependencies()

		for epoch in range(epochs):
			train_loss = self.__train_step(train_loader)
			valid_loss = self.__valid_step(valid_loader)
			best_valid_loss = self.__save_model(valid_loss, best_valid_loss)

			if self.__verbose:
				message = 'Epoch {} losses: {} (train)\t{} (valid)'
				print(message.format(epoch, train_loss, valid_loss))

	def predict(self, x):
		self.__verify_predict_dependencies()
		probabilities = F.softmax(self.__best_model(x))
		return probabilities.argmax(dim=1)

	def score(self, x, y):
		yhat = self.predict(x)
		accuracy = torch.sum(y == yhat)
		return accuracy

	def score_loader(self, loader):
		score = 0
		for x, y in loader:
			score += self.score(x, y) / len(loader)
		return score
