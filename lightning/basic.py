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
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
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
    num_epochs = 1

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(os.getcwd(), download=True, train=True, transform=transform)
    test_set = datasets.MNIST(os.getcwd(), download=True, train=False, transform=transform)

    train_set_size = int(len(train_set) * split_rate)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set)
    valid_loader = DataLoader(valid_set)
    test_loader = DataLoader(test_set)

    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    trainer = L.Trainer(accelerator=devices, devices=1, max_epochs=epochs)
    trainer.fit(autoencoder, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader)