import torch
from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = nn.Linear(7*7*64,128)
        self.linear_2 = nn.Linear(128,10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0),-1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = F.log_softmax(self.linear_2(x),dim=1)

        return pred


