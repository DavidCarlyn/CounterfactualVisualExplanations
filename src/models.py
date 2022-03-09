import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=0, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=0, stride=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(in_features=180, out_features=360)
        self.linear2 = nn.Linear(in_features=360, out_features=10)


    def forward(self, x):
        x = self.extract_features(x)
        y = self.predict(x)

        return x, y

    def extract_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        return x


    def predict(self, x):
        x = x.view(x.shape[0], -1)
        y = self.relu(self.linear1(x))
        y = self.relu(self.linear2(y))
        return y