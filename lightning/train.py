import os

from argparse import ArgumentParser

import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelSummary, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"]) # if want to save all the hyperparameters
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)

        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--devices", type=str, default="gpu") # or cpu
    parser.add_argument("--epochs", type=int, default=1) # or cpu

    args = parser.parse_args()
    devices = args.devices
    epochs = args.epochs

    seed = torch.Generator().manual_seed(42)
    split_rate = 0.8

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(os.getcwd(), download=True, train=True, transform=transform)
    test_set = datasets.MNIST(os.getcwd(), download=True, train=False, transform=transform)

    train_set_size = int(len(train_set) * split_rate)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, num_workers=7)
    valid_loader = DataLoader(valid_set, num_workers=7)
    test_loader = DataLoader(test_set, num_workers=7)

    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    # basic
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs)

    # fast_dev_run: The trainer runs 5 batch of train, val, test, predict(True for 5, any number is possible)
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, fast_dev_run=True)

    # limit_train_batches, limit_val_batches: shorten the epoch length
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, limit_train_batches=0.1, limit_val_batches=0.01)
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, limit_train_batches=10, limit_val_batches=5)

    # num_sanity_val_steps: Lightning runs 2 steps of validation in the beginning of training.
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, num_sanity_val_steps=2)

    # weight summary: add the child modules to the summary
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, callbacks=[ModelSummary(max_depth=-1)])

    # profiling: Once the .fit() function has completed, the profile measures.
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, limit_train_batches=0.1, limit_val_batches=0.5, profiler="simple")

    # DeviceStatsMonitor ensures that you’re using the full capacity of your accelerator (GPU/TPU/IPU/HPU).
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, limit_train_batches=0.1, limit_val_batches=0.1, callbacks=[DeviceStatsMonitor()])
     
    # wandb logger
    wandb_logger = WandbLogger(log_model=True)
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, limit_train_batches=0.1, limit_val_batches=0.1, logger=wandb_logger)

    # set the path of the checkpoint
    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs, default_root_dir="./lightning_logs")
    """
    In a Lightning checkpoint,
    - 16-bit scaling factor (if using 16-bit precision training)
    - Current epoch
    - Global step
    - LightningModule’s state_dict
    - State of all optimizers
    - State of all learning rate schedulers
    - State of all callbacks (for stateful callbacks)
    - State of datamodule (for stateful datamodules)
    - The hyperparameters (init arguments) with which the model was created
    - The hyperparameters (init arguments) with which the datamodule was created
    - State of Loops
    """

    # basic
    trainer.fit(autoencoder, train_loader, valid_loader)

    # automatically restores model, epoch, step, LR schedulers, etc..., i.e., restore the full training
    # trainer.fit(autoencoder, ckpt_path="some/path/to/my_checkpoint.ckpt")

    trainer.test(dataloaders=test_loader, ckpt_path="./lightning_logs/version_0/checkpoints/epoch=0-step=48000.ckpt")