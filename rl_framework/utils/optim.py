from torch import optim
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr_scheduler


class OptimizerManager:
    def __init__(self, learning_rate: float, **reduce_lr_on_plateau_options):
        self.init_lr = learning_rate
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **reduce_lr_on_plateau_options
        )

    def __call__(self, parameters):
        self.optimizer = optim.Adam(parameters, lr=self.lr)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_lr(self, new_lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    def step_scheduler(self, metric):
        self.scheduler.step(metric)


def get_lr(optimizer: Optimizer):
    return optimizer.param_groups[0]["lr"]
