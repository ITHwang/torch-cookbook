import torch
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

class Model1(DemoModel):
    def configure_optimizers(self):
        print("using Model1...")
        return super().configure_optimizers()

class Model2(DemoModel):
    def configure_optimizers(self):
        print("using Model2...")
        return super().configure_optimizers()
