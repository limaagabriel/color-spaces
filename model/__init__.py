from abc import ABC


class Classifier(ABC):
	def __init__(self, *args, **kwargs):
		self.__model = self.module(*args, **kwargs)
		self.__optimizer = None
		self.__criterion = None

	def set_optimizer(self, optimizer, **kwargs):
		self.__optimizer = optimizer(self.__model.parameters(), **kwargs)

	def set_loss_criterion(self, criterion, **kwargs):
		self.__criterion = criterion(**kwargs)

	def __verify_dependencies(self):
		if self.__optimizer is None:
			raise ValueError('The optimizer must be defined before this operation')
		if self.__criterion is NOne:
			raise ValueError('The loss criterion must be defined before this operation')

	def fit(self, train_loader, valid_loader=None):
		self.__verify_dependencies()
