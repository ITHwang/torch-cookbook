import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from torch.optim.optimizer import Optimizer as Optimizer

import models # noqa: F401
import data_modules # noqa: F401
import optimizers # noqa: F401

# base
# python main.py fit --model.learning_rate 0.1
# python main.py fit --model.out_dim 10 --model.learning_rate 0.1
# python main.py fit --model.out_dim 2 --model.learning_rate 0.1 --data.data_dir '~/' --trainer.logger False
cli = LightningCLI(DemoModel, BoringDataModule)

# multiple models
# python main.py fit --model Model1
# python main.py fit --model Model2
cli = LightningCLI(datamodule_class=BoringDataModule)

# multiple data modules
# python main.py fit --data FakeDataset1
# python main.py fit --data FakeDataset2
cli = LightningCLI(DemoModel)

# python main.py fit --optimizer LitAdam
# python main.py fit --optimizer FancyAdam
cli = LightningCLI(DemoModel, BoringDataModule)

# python main.py fit --optimizer=Adam --lr_scheduler CosineAnnealingLR
cli = LightningCLI(DemoModel, BoringDataModule)

# python main.py fit --config config.yaml

"""
# TIP
The options that become available in the CLI are the __init__ parameters of the LightningModule and LightningDataModule classes.
Thus, to make hyperparameters configurable, just add them to your class’s __init__.
"""