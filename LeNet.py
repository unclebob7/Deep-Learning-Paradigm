import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        """
        dynamic dataflow, 
        here we consider non-linearity as function instead of layer
        """
        pool1_out = self.pool1(F.relu(self.conv1(x)))
        pool2_out = self.pool2(F.relu(self.conv2(pool1_out)))
        flatten = pool2_out.view(-1, 16 * 5 * 5)
        fc1_out = F.relu(self.fc1(flatten))
        fc2_out = F.relu(self.fc2(fc1_out))
        logits = self.fc3(fc2_out)
        return logits
