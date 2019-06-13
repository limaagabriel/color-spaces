import sys
from abc import ABC
from abc import abstractmethod


class StopCriterionDefinition(ABC):
	def __init__(self):
		self.__iterations = 0

	def initialize(self):
		self.__iterations = 0

	@property
	def iterations(self):
		return self.__iterations

	def __call__(self, valid_loss):
		self.__iterations += 1
		return self.check(self.__iterations, valid_loss)

	@abstractmethod
	def check(self, iterations, valid_loss):
		pass


class StopCriterion(object):
	@staticmethod
	def iteration_limit(max_iterations):
		class IterativeStopCriterion(StopCriterionDefinition):
			def check(self, iterations, valid_loss):
				return iterations > max_iterations

		return IterativeStopCriterion()

	@staticmethod
	def valid_loss(min_loss, max_iterations=sys.maxsize):
		class ValidationLossStopCriterion(StopCriterionDefinition):
			def check(self, iterations, valid_loss):
				return valid_loss < min_loss or iterations > max_iterations

		return ValidationLossStopCriterion()
