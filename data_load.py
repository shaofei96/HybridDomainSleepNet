# coding:utf-8
import torch
from torch.utils.data import DataLoader, TensorDataset
from Index_calculation import *
import numpy as np
from DataGenerator import *
from MASS_SS3_Utils import *
import os
import pandas as pd

batch_size1 = 256
batch_size2 = 256

fold = 1

train_data = np.load(rf"G:\sleep data\MASS SS3\AddContext_data\Validation_fold_{fold}_1\\train_data.npy")
train_labels = np.load(rf"G:\sleep data\MASS SS3\AddContext_data\Validation_fold_{fold}_1\\train_labels.npy").reshape(-1, 1)

train_data = torch.tensor(train_data, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.float).squeeze_(1)

test_data = np.load(rf"G:\sleep data\MASS SS3\AddContext_data\Validation_fold_{fold}_1\\test_data.npy")
test_labels = np.load(rf"G:\sleep data\MASS SS3\AddContext_data\Validation_fold_{fold}_1\\test_labels.npy").reshape(-1, 1)


test_data = torch.tensor(test_data, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.float).squeeze_(1)

# print(train_data.shape)
# print(train_labels.shape)
# print(test_data.shape)
# print(test_labels.shape)


trainData =TensorDataset(train_data, train_labels)
testData =TensorDataset(test_data, test_labels)

train_dataloader = DataLoader(trainData, batch_size=batch_size1, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size2, shuffle=True, drop_last=True)

print("----------!!!数据加载完毕!!!----------")

