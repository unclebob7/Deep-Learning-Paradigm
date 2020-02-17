import os
import torch
import torchvision
import torchvision.transforms as transforms
from hparams import Hparams

PATH = "./data"
DOWNLOAD = not os.path.exists(PATH)

# data transformation set
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0, 0, 0), (1, 1, 1))])

# training and test set
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True,
                                         download=DOWNLOAD, 
                                         transform=transform)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=4,
                                           shuffle=True)

val_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=False,
                                         download=DOWNLOAD, 
                                         transform=transform)

val_loader = torch.utils.data.DataLoader(val_set,
                                          batch_size=len(val_set),
                                          shuffle=False)