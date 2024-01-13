import torch.nn as nn
import json
import os

with open("config.json", "r") as config_file:
    config = json.load(config_file)

data_path_main = config["data_path_main"]
inputChannels = config["inputChannels"]

classes = [d for d in os.listdir(data_path_main) if os.path.isdir(os.path.join(data_path_main, d))]
numClasses = len(classes)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=numClasses):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.relu3 = nn.ReLU()  # Değişiklik burada
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 128 * 56 * 56)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

