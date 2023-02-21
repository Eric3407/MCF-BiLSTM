import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.models import MCF_BiLSTM
from utils.dataset import ROIDataset, get_label
from utils.training import train_model, accuracy, compute_accuracy
import random
import numpy as np

from torchsummary import summary
import matplotlib.pyplot as plt
import pylab as pl





if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print('Current device is {}'.format(device))

    batch_size = 200;

    model = MCF_BiLSTM().to(device)

    test_dataset = ROIDataset(path='data/test', key=get_label, mode='classification', gen_p=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model.eval()

    model.load_state_dict(torch.load('data/MCF_BiLSTM'))

    with torch.no_grad():
        test_acc=compute_accuracy(model, test_loader, device)

    print('test_accuracy: {:.4f}'.format(test_acc))

#
def confusion_matrix(preds, labels, conf_matrix):

    for p, t in zip(preds, labels):
        conf_matrix[p.long(), t.long()] += 1
    return conf_matrix

all_preds=torch.tensor([]).to(device)
all_labels=torch.tensor([]).to(device)
conf_matrix = torch.zeros(3, 3)
with torch.no_grad():

    for x,y in test_loader:
        out=model(x.to(device))
        labels=y.to(device)
        _, preds = torch.max(out, 1)
        all_labels=torch.cat((all_labels,labels),dim=0)
        all_preds=torch.cat((all_preds,preds),dim=0)


print(all_labels)
print(all_preds)
conf_matrix = confusion_matrix(all_preds, all_labels, conf_matrix)
conf_matrix = conf_matrix.cpu()


conf_matrix = np.array(conf_matrix.cpu())
per_kinds = conf_matrix.sum(axis=0)
corrects = conf_matrix.diagonal(offset=0)
print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), 600))
print(conf_matrix)


print("每种ROI总个数：", per_kinds)
print("每种ROI预测正确的个数：", corrects)
print("每种ROI的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

labels = ['1', '2', '3']
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

thresh = conf_matrix.max() / 2
class_kinds = 3
for x in range(class_kinds):
    for y in range(class_kinds):

        info = int(conf_matrix[y, x])
        plt.text(x, y, info,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if info > thresh else "black")

plt.tight_layout()
plt.yticks(range(class_kinds), labels)
plt.xticks(range(class_kinds), labels, rotation=0)
plt.xlabel('Ground truth',fontdict={'family' : 'Times New Roman', 'size': 14})
plt.ylabel('Prediction',fontdict={'family' : 'Times New Roman', 'size': 14})

plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.savefig(fname="con_matrix.svg",format="svg",bbox_inches='tight')
plt.show()









