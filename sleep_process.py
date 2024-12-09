import numpy as np
import os
import pandas as pd
from mne.io import read_raw_edf
from DE import DE


def AddContext(x, context, label=False, dtype=float):
    assert context % 2 == 1, "context value error."

    cut = int(context / 2)
    if label:
        tData = x[cut:x.shape[0] - cut]
        # print(tData.shape)
    else:
        tData = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=dtype)
        for i in range(cut, x.shape[0] - cut):
            tData[i - cut] = x[i - cut:i + cut + 1]
        # print(tData.shape)
    return tData

# the parameters to extract DE and PSD
stft_para = {
    'stftn': 6000,  # 30*Fs; where Fs = 200
    'fStart': [0.5, 2, 4, 6, 8, 11, 14, 22, 31],
    'fEnd': [4, 6, 8, 11, 14, 22, 31, 40, 50],
    'fs': 200,
    'window': 30,
}

context = 9

data_path = r"G:\sleep data\BP SLEF sleep\signals_edf\\"
labels_path = r"G:\sleep data\BP SLEF sleep\labels_csv\T3\\"
data_list = os.listdir(data_path)
data_list.sort(key=lambda x: int(x[6:-4]))
labels_list = os.listdir(labels_path)
labels_list.sort(key=lambda x: int(x[6:-14]))
# print(labels_list)
# i = 0
for i in range(len(data_list)):
    raw = read_raw_edf(data_path + data_list[i])
    labels = np.array(pd.read_csv(labels_path + labels_list[i], header=None))  # (1059, 1)

    raw_data, times = raw[:, :]  # (33, 6358800)
    raw_data = raw_data[:, :len(labels) * 30 * 200].reshape(33, -1, 30 * 200).transpose(1, 0, 2)  # (labels, 33, 6000)

    raw_data_list=raw_data[:, :33, :]

    MYpsd = np.zeros([raw_data_list.shape[0], raw_data_list.shape[1], len(stft_para['fStart'])], dtype=float)
    MYde = np.zeros([raw_data_list.shape[0], raw_data_list.shape[1], len(stft_para['fStart'])], dtype=float)
    for j in range(0, raw_data_list.shape[0]):
        data = raw_data_list[j]
        # print("raw_data_list:", data.shape)
        MYde[j] = DE(data, stft_para)

    print(MYde.shape, end=' ')

    MYde = AddContext(MYde, context)
    labels = AddContext(labels, context, label=True)

    print(MYde.shape, end=' ')
    print(labels.shape, end=' ')

    save_dir = rf"G:\sleep data\BP SLEF sleep\AddContext_data_T3\S{i+1}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.save(rf"G:\sleep data\BP SLEF sleep\AddContext_data_T3\S{i+1}\data.npy", MYde)
    np.save(rf"G:\sleep data\BP SLEF sleep\AddContext_data_T3\S{i+1}\labels.npy", labels)