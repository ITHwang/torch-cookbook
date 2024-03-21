import os

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

import lightning as L

from train import Encoder, Decoder, LitAutoEncoder

model = LitAutoEncoder.load_from_checkpoint("lightning_logs/version_9/checkpoints/epoch=0-step=4800.ckpt", encoder=Encoder(), decoder=Decoder())
model.eval()

transform = transforms.ToTensor()
test_set = datasets.MNIST(os.getcwd(), download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, num_workers=7)

trainer = L.Trainer(accelerator="gpu", devices=1)
predictions = trainer.predict(model, test_loader)

# Get the predicted labels
predicted_labels = [torch.argmax(pred).item() for pred in predictions]

# Get the true labels
true_labels = [label for _, label in test_loader]

# Calculate the accuracy
accuracy = torch.tensor(predicted_labels).eq(torch.tensor(true_labels)).float().mean().item()
print(f"Accuracy: {accuracy}")