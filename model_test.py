import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Model import *
import warnings
import matplotlib.pyplot as plt
from Index_calculation import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.manifold import TSNE
import pandas as pd
import mne
import scipy.io as sio


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

def cm_plot_number(original_label, predict_label):
    cm = confusion_matrix(original_label, predict_label)

    cm_new = np.zeros(shape=[5, 5])
    for x in range(5):
        t = cm.sum(axis=1)[x]
        for y in range(5):
            cm_new[x][y] = round(cm[x][y] / t * 100, 2)

    plt.matshow(cm_new, cmap=plt.cm.Blues)

    plt.colorbar()
    x_numbers = []
    y_numbers = []
    for x in range(5):
        y_numbers.append(cm.sum(axis=1)[x])
        x_numbers.append(cm.sum(axis=0)[x])
        for y in range(5):
            percent = format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f")

            plt.annotate(format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f"), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')

    y_stage = ["W\n(" + str(y_numbers[0]) + ")", "N1\n(" + str(y_numbers[1]) + ")", "N2\n(" + str(y_numbers[2]) + ")",
               "N3\n(" + str(y_numbers[3]) + ")", "REM\n(" + str(y_numbers[4]) + ")"]
    x_stage = ["W\n(" + str(x_numbers[0]) + ")", "N1\n(" + str(x_numbers[1]) + ")", "N2\n(" + str(x_numbers[2]) + ")",
               "N3\n(" + str(x_numbers[3]) + ")", "REM\n(" + str(x_numbers[4]) + ")"]
    y = [0, 1, 2, 3, 4]
    plt.xticks(y, x_stage)
    plt.yticks(y, y_stage)
    # sns.heatmap(cm_percent, fmt='g', cmap="Blues", annot=True, cbar=False, xticklabels=x_stage, yticklabels=y_stage)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒

    plt.tight_layout()
    # plt.savefig("%smatrix%s.svg" % (savepath, str(knum)), bbox_inches='tight')  ##bbox_inches用于解决显示不全的问题
    # plt.show()
    plt.show()
    plt.close()
    # plt.savefig("/home/data_new/zhangyongqing/flx/pythoncode/"+str(knum)+"matrix.jpg")
    return kappa(cm), f1_score(original_label,predict_label,average=None), cm

def divide_data_based_on_labels(datas, labels):

    # print(datas.shape)
    # print(labels.shape)

    index_dict = {}
    datas_dict = {}

    # print(labels)
    for i, num in enumerate(labels):
        # new_num = one_hot_to_number(num)
        num = num.item()
        if num not in index_dict:

            index_dict[int(num)] = [i]
            datas_dict[int(num)] = [datas[i,:]]
            # print("aaa")
        else:
            index_dict[int(num)].append(i)
            datas_dict[int(num)].append(datas[i, :])
            # print("bbb")


    unique_labels = list(index_dict.keys())
    return datas_dict, unique_labels

def tSNE_2D(datas, labels, label_names):
    datas = datas.cpu().detach().numpy()
    labels = labels.cpu()
    # print(type(datas))
    datas = datas.reshape(datas.shape[0], -1)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    datas_tsne = tsne.fit_transform(datas)
    # print(type(datas_tsne))

    # print(datas_tsne.shape)

    datas_tsne, unique_labels = divide_data_based_on_labels(datas_tsne, labels)
    print(datas_tsne)
    print(np.array(unique_labels).shape)

    # datas_tsne = np.array(datas_tsne[1])
    # print(datas_tsne[:,0])

    markers = ['*', 'o', 'v', 'd', 'x']
    # markers = ['o', 'o', 'o', 'o', 'o']
    # markers = ['.', '.', '.', '.', '.']

    # Plot the data in 2D
    plt.figure(figsize=[15, 10])
    plt.xticks([])
    plt.yticks([])
    index = 0
    for label in unique_labels:
        print(label)
        datas_array_tsne = np.array(datas_tsne[label])
        print(label_names)
        plt.scatter(datas_array_tsne[:, 0], datas_array_tsne[:, 1], marker=markers[index])
        index += 1
    # plt.title(title)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.legend(prop={'family' : 'Times New Roman', 'size' : 12}, labels=['N3', 'N2', 'N1', 'REM', 'Wake'])
    plt.savefig('C:/Users/WP666/Desktop/TSNE_BP_SLEF_T2_1.svg', format='svg', dpi= 120, bbox_inches='tight')
    plt.show()

def tSNE_3D(datas, labels,title, label_names):
    datas = datas.cpu().detach().numpy()
    labels = labels.cpu()
    # print(type(datas))
    # print(datas.shape)
    datas = datas.reshape(datas.shape[0], -1)


    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=4, random_state=42)
    datas_tsne = tsne.fit_transform(datas)

    # print(type(datas_tsne))

    # print(datas_tsne.shape)

    datas_tsne, unique_labels = divide_data_based_on_labels(datas_tsne,labels)

    # datas_tsne = np.array(datas_tsne[1])
    # print(datas_tsne[:,0])

    markers = ['.','o','d','*','v']
    # markers = ['o', 'o', 'o', 'o', 'o']

    # Plot the data in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    index = 0
    for label in unique_labels:
        print(label)
        datas_array_tsne = np.array(datas_tsne[label])
        ax.scatter(datas_array_tsne[:, 0], datas_array_tsne[:, 1], datas_array_tsne[:, 2], label=label_names[label], marker=markers[index])

        index += 1

    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.legend()
    plt.show()


batch_size1 = 256
batch_size2 = 256

fold = 18

# test_data = np.load(rf"G:\sleep data\BP SLEF sleep\Folds_T3\ExS{fold}\\test_data.npy").transpose(0, 2, 1, 3)
# test_labels = np.load(rf"G:\sleep data\BP SLEF sleep\Folds_T3\ExS{fold}\\test_labels.npy").reshape(-1, 1)

# test_data = np.load(rf"G:\sleep data\BP SLEF sleep\Folds\ExS{fold}\\test_data.npy").transpose(0, 2, 1, 3)
# test_labels = np.load(rf"G:\sleep data\BP SLEF sleep\Folds\ExS{fold}\\test_labels.npy").reshape(-1, 1)

test_data = np.load(rf"G:\sleep data\MASS SS3\AddContext_data\Validation_fold_{fold}_1\\test_data.npy")
test_labels = np.load(rf"G:\sleep data\MASS SS3\AddContext_data\Validation_fold_{fold}_1\\test_labels.npy").reshape(-1, 1)

test_data = torch.tensor(test_data, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.float).squeeze_(1)

print("----------!!!数据加载完毕!!!----------")

learning_rate = 0.03
epochs = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

myModel = model().to(device)
myModel = torch.load(
    rf"D:\JetBrains\PyCharm Community Edition 2019.2.4\project\SleepMonitor\model_weight\model_fold_{fold}.pth",
    map_location=lambda storage, loc: storage.cuda(0))

# myModel = torch.load(
#     rf"G:\sleep data\BP SLEF sleep\Model_Weight_T3\model_Brain_pic_S{fold}.pth",
#     map_location=lambda storage, loc: storage.cuda(0))

# myModel = torch.load(
#     rf"G:\sleep data\BP SLEF sleep\Model_Weight_T3\model_S{fold}.pth",
#     map_location=lambda storage, loc: storage.cuda(0))

# myModel = torch.load(
#     r"D:\JetBrains\PyCharm Community Edition 2019.2.4\project\SleepMonitor\model_weight\model_brain_pic_S9.pth",
#     map_location=lambda storage, loc: storage.cuda(0))

# myModel = torch.load(
#     rf"G:\sleep data\BP SLEF sleep\Modle_Weight\model_S{fold}.pth",
#     map_location=lambda storage, loc: storage.cuda(0))

G = testclass()

test_len = G.len(test_data.shape[0], batch_size2)

test_acc_plt = []

Test_Accuracy_list = []

total_test_acc = 0
total_test_step = 1

X_test = test_data.to(device)
Y_test = test_labels.to(device)

# output, output1 = myModel(X_test)
output = myModel(X_test)

output = output.cpu().detach().numpy()
Y_test = Y_test.cpu().detach().numpy()

sio.savemat('BPoutput_1.mat',mdict={'out':output})
sio.savemat('BPYtest_1.mat',mdict={'Y_t':Y_test})


# test_acc = (output.argmax(dim=1) == Y_test).sum()
# # TP_TN_FP_FN = G.Compute_TP_TN_FP_FN(test_label, label, matrix)
#
# total_test_step = total_test_step + 1
#
# test_acc_plt.append(test_acc)
# total_test_acc += test_acc

# # 计算k、c、f
# print("Test:", total_test_acc / test_len)
# kappa, f1, cm = cm_plot_number(Y_test.cpu().detach().numpy().squeeze(), np.argmax(output.cpu().detach().numpy(), axis=1).squeeze())
# print("kappa:", [kappa])
# print("cm:", cm)
# print("f1_score:", f1)

# print("----------------------------------------------------------------------------------------------------")

# TSNE
tSNE_2D(X_test, test_labels, "01234")
# tSNE_2D(output, test_labels, "01234")

# # 脑地形图
# # 图示电极位置
# output1 = output1.cpu().detach().numpy()
# # output1 = output1.reshape(output1.shape[0], -1)[:, 2:22]
# output1 = output1.reshape(output1.shape[0], -1)[:, 0:30]
# # print(output1.shape)
# data1020 = pd.read_excel('BP.xlsx', index_col=0)
# channels1020 = np.array(data1020.index)
# value1020 = np.array(data1020)
# list_dic = dict(zip(channels1020, value1020))
# montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
#                                              nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
#                                              lpa=[-0.08609924, -0., -0.04014873],
#                                              rpa=[0.08609924, 0., -0.04014873])
# info = mne.create_info(ch_names=montage_1020.ch_names, sfreq=200, ch_types='eeg')
# # 图示电极位置
# # montage_1020.plot()
# # plt.show()
#
# for i in range(800):
#
#     # plt.subplot(1, 1, i + 1)
#     evoked = mne.EvokedArray(output1[i,:].reshape(30, 1), info)
#     # print(eeg_re_prm[:, i].reshape(31, 1))
#     # print()
#     evoked.set_montage(montage_1020)
#     im, cm = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=False, cmap='viridis')
#
#     plt.title("")
#     # 添加所有子图的colorbar
#
#     # plt.colorbar(im)
#     plt.tight_layout()
#     # plt.show()
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     # name = os.path.splitext(eeg_re_prm)[0]
#     plt.savefig(rf"D:/JetBrains/PyCharm Community Edition 2019.2.4/project/SleepMonitor/figures/{i}.svg", format='svg', dpi=150)
#
#
#     plt.close()

