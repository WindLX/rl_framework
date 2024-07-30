from abc import ABCMeta, abstractmethod

from numpy import ndarray


class Operation(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: ndarray) -> ndarray:
        raise NotImplementedError("Each operation must implement the forward method.")


class Pipeline:
    def __init__(self, *operations: Operation):
        self.operations = list(operations)

    def add_operation(self, operation: Operation):
        self.operations.append(operation)

    def forward(self, x: ndarray) -> ndarray:
        for operation in self.operations:
            x = operation.forward(x)
        return x

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)
