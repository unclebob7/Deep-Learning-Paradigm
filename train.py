import logging
import math
from hparams import Hparams
from utils import save_hparams, latest_ckpt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import train_set, train_loader, val_loader
from LeNet import LeNet
from utils import view_bar
from sklearn.metrics import accuracy_score
from validation import validation
import time

# load the hyper-parameters
logging.basicConfig(level=logging.INFO)
logging.info("# Loading hyperparameters")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.train_dir)

# identify the device to use
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using %s" % DEVICE)

# instantiate the model, the loss function and optimizer
model = LeNet()
xentropy = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=hp.lr,
                      momentum=0.9)

# check the latest checkpoint to resume training
ckpt_path = latest_ckpt(hp.ckpt)
if ckpt_path is None:
    logging.info("Initializing from scratch")
    epoch_start = 1
else:
    # resume training
    logging.info("Loading the latest checkpoint")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch_start = ckpt['epoch_start']
    model.train()

# Load model to device
logging.info("# Load model to %s" % (DEVICE))
model = model.to(DEVICE)

# training
logging.info("# Start training")
start_time = time.time()
for epoch in range(epoch_start, hp.epochs + 1):
    num_batch = math.floor(len(train_set) / hp.batch_size)
    for i, data in enumerate(train_loader, 0):
        # data comes in the form of [src, target]
        src, target = data[0].to(DEVICE), data[1].to(DEVICE)
        # feed forward
        predicts = model(src)
        loss = xentropy(predicts, target)
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # validation
    with torch.no_grad():
        accur = validation(DEVICE, model)
    end_time = time.time()
    elapse = end_time - start_time
    # console visualization
    view_bar("training: ", epoch, hp.epochs, accur, elapse)
    torch.save({"epoch_start": epoch,
                "model_state_dict": model.state_dict()}, "./ckpt/ckpt_%s.pth" % (epoch))


