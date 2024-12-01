
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import ToTensor

imageTransformer = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)),
    ToTensor(),
    transforms.Normalize(0,1)
])

image_dataset = datasets.ImageFolder(
    root='data',
    transform=imageTransformer
)

train_dataset, test_dataset = random_split(image_dataset, [0.8, 0.2])

train_dataset_loader = DataLoader(train_dataset, shuffle=True)

classes = image_dataset.find_classes('data')

class TuberculosisDetector(nn.Module):
    def __init__(self):
        super(TuberculosisDetector, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,100, kernel_size=(3,3), stride=1),
            nn.MaxPool2d((2,2), stride=(1,1)),
            nn.ReLU(),

            nn.Conv2d(100, 50, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(50, 20, kernel_size=(3, 3), stride=1),
            nn.MaxPool2d((3, 3), stride=(2, 2)),
            nn.ReLU(),

            nn.BatchNorm2d(num_features=20),

            nn.Flatten(),

            nn.Linear(in_features=9680, out_features=2000),
            nn.ReLU(),

            nn.Linear(in_features=2000, out_features=500),
            nn.ReLU(),

            nn.Linear(in_features=500, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.layers(x)
        return y

model = TuberculosisDetector()

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epochs = 2):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        count = 0
        for step, (x, y) in enumerate(train_dataset_loader):
            optimizer.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y.unsqueeze(1).to(torch.float))

            loss.backward()
            optimizer.step()
            running_loss += loss
            count += 1
            print(f"Running loss: {running_loss / count}")


# train(2)
#
# torch.save(model.state_dict(), "trained_model.dict")