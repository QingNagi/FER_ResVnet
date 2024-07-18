import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from resVnet_cbam import resVnet18_CBAM
from ckplus_data import data_pre
from eca_channel_resnet import ResNet18_ace
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASS_LABELS = ['anger', 'disgusted', 'fearul', 'happy', 'neutral', 'sad', "surprised"]

model = resVnet18_CBAM().to(device)
model.load_state_dict(torch.load(r'CBAM_all_last_resnet_best_73.73.pth'))
val_db = data_pre(r'D:\pythonProject3\FER2013\FER2013\PublicTest', 40, mode='val')
val_loader = DataLoader(val_db, batch_size=128, num_workers=1)


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    # for x, y, z in loader:
    for x, y in loader:
        # x, y = x.to(device), y.to(device)
        x_1, x_2, y = x[:, :6, :, :, :].to(device), x[:, 6:, :, :, :].to(device), y.to(device)
        bs, nc, c, h, w = x_1.shape
        x_1 = x_1.view(-1, c, h, w)
        x_2 = x_2.view(-1, c, h, w)
        with torch.no_grad():  # 无梯度环境 不用更新
            # logits = model(x, z)
            logits_1 = model(x_1)
            logits_2 = model(x_2)
            logits = (logits_2 + logits_1)
            logits_vag = logits.view(bs, nc, -1).mean(1)
            pred = logits_vag.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


if __name__ == '__main__':
    val_acc = evalute(model, val_loader)
    print(val_acc)


