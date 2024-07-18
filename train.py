import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader
from ckplus_data import data_pre
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import numpy as np
from resVnet_cbam import resVnet18_CBAM

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()  # 释放显存

batchsz = 128
epochs = 240

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = data_pre(r'D:\pythonProject3\FER2013\fer2013plus\fer2013\train', 40, mode='train')
val_db = data_pre(r'D:\pythonProject3\FER2013\fer2013plus\fer2013\test', 40, mode='val')


train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=1)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=1)

viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()
    '''model = create_RepVGG_B0(deploy=True).to(device)
    model.load_state_dict(torch.load(r'D:\pythonProject3\FER2013\model_last\repVgg_FER.pth'))'''

    correct = 0
    total = len(loader.dataset)

    # for x, y, z in loader:
    for x, y in loader:
        # x, y = x.to(device), y.to(device)
        x_1, x_2, y = x[:, :6, :, :, :].to(device), x[:, 6:, :, :, :].to(device), y.to(device)
        bs, nc, c, h, w = x_1.shape
        x_1 = x_1.view(-1, c, h, w)
        x_2 = x_2.view(-1, c, h, w)
        # x = x.view(-1, c, h, w)
        with torch.no_grad():  # 无梯度环境 不用更新
            # logits = model(x, z)
            logits_1 = model(x_1)
            logits_2 = model(x_2)
            logits = (logits_2 + logits_1)
            logits_vag = logits.view(bs, nc, -1).mean(1)
            pred = logits_vag.argmax(dim=1)
            # pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)
    return true_dist.to(device)


def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)


def main():
    lr = 0.1
    model = resVnet18_CBAM().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 - 1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10)
    # criteon = nn.CrossEntropyLoss()
    # criteon = CCCLoss()
    criteon = cross_entropy

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    since = time.time()
    for epoch in range(epochs):
        # torch.cuda.empty_cache()  # 释放显存
        train_loss = 0.0
        train_num = 0
        # for step, (x, y, z) in enumerate(train_loader):
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            # x, y, z = x.to(device), y.to(device), z.to(device)
            x, y = x.to(device), y.to(device)
            bs, nc, c, h, w = x.shape
            x = x.view(-1, c, h, w)
            y = torch.repeat_interleave(y, repeats=nc, dim=0)

            model.train()
            # logits = model(x, z)
            x, targets_a, targets_b, lam = mixup_data(x, y)
            soft_labels_a = smooth_one_hot(
                targets_a, classes=7, smoothing=0.1)
            soft_labels_b = smooth_one_hot(
                targets_b, classes=7, smoothing=0.1)
            # soft_label = smooth_one_hot(y, classes=7, smoothing=0.1)
            logits = model(x)
            # y = F.one_hot(y, num_classes=8).float()  # n为类别数
            loss = mixup_criterion(criteon, logits, soft_labels_a, soft_labels_b, lam)
            # loss = criteon(logits, y)
            # loss = criteon(logits, soft_label)

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), max_norm=10)
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

        # repvgg_model_convert(model, save_path=r'D:\pythonProject3\FER2013\model_last\repVgg_FER.pth')
        viz.line([train_loss / train_num], [global_step], win='loss', update='append')
        global_step += 1

        print('epoch:{}, lr:{}'.format(epoch+1, lr))
        time_use = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))

        if epoch % 1 == 0:
            val_acc = evalute(model, val_loader)
            # Update learning rate
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), r'D:\pythonProject3\FER2013\model_last\CBAM_last_smooth_align_best.pth')
                # repvgg_model_convert(model, save_path=r'D:\pythonProject3\FER2013\model_last\CBAM_last_ViT_best.pth')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
            if best_acc > 0.7:
                lr = 0.05

    print('best acc:', best_acc, 'best epoch:', best_epoch)

    # model.load_state_dict(torch.load(r'D:\pythonProject3\alexnet_1\model\CCC_model.pkl'))  # 用最好的状态覆盖模型
    print('loaded from ckpt!')


if __name__ == '__main__':
    main()
