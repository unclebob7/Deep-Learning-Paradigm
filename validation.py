import torch
import numpy as np
from data_preprocessing import val_loader
from sklearn.metrics import accuracy_score

def validation(DEVICE, model):
    for j, val_data in enumerate(val_loader, 0):
        val_src, val_target = val_data[0].to(DEVICE), val_data[1].to(DEVICE)
        predicts = model(val_src)
        accur = accuracy_score(val_target.detach().numpy(), np.argmax(predicts.detach().numpy(), axis=1))
        return accur