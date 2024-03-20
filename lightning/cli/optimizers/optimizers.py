import torch
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

class LitAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using LitAdam", "⚡")
        super().step(closure)

class FancyAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using FancyAdam", "⚡")
        super().step(closure)

class LitLRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
    def step(self):
        print("⚡", "using LitLRScheduler", "⚡")
        super().step()
