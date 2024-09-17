import torch
import glob
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random, csv

from torchvision.transforms.functional import crop, pad
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import visdom
import time
import torchvision
import face_alignment
import cv2
import numpy as np
from ast import literal_eval
from cutout import Cutout
warnings.filterwarnings("ignore")

viz = visdom.Visdom()

def crop_left_upper(image):
    crops = crop(image, 0, 0, 40, 40)
    return

class Pokemon1(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon1, self).__init__()
        self.model = mode
        self.root = root
        self.resize = resize

        self.name2label = {}  # 'sq...'
        for name in sorted(os.listdir((os.path.join(root)))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)
        # image, label
        # self.images, self.labels, self.landmarks = self.load_csv('images2.csv')
        self.images, self.labels = self.load_csv('images1.csv')
        '''if mode == 'train':
            self.images = self.images[:int(0.9 * len(self.images))]
            self.labels = self.labels[:int(0.9 * len(self.labels))]
            # self.landmarks = self.landmarks[:int(0.75 * len(self.landmarks))]
        elif mode == 'val':
            self.images = self.images[int(0.9 * len(self.labels)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]'''
            # self.landmarks = self.landmarks[int(0.75 * len(self.landmarks)):]
        '''elif mode == 'test':
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]'''
            # self.landmarks = self.landmarks[:]
        if mode == 'test':
            self.images = self.images[:]
            self.labels = self.labels[:]
            #self.landmarks = self.landmarks[:int(0.5 * len(self.landmarks))]
        elif mode == 'val':
            self.images = self.images[:]
            self.labels = self.labels[:]
            #self.landmarks = self.landmarks[int(0.5 * len(self.landmarks)):]
        elif mode == 'train':
            self.images = self.images[:]
            self.labels = self.labels[:]
            #self.landmarks = self.landmarks[:]

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))
            print(len(images), images)
            # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    # input = cv2.imread(img)
                    # det = fa.get_landmarks_from_image(input)
                    # det = np.array(det, dtype=int)[0, :, :].T
                    # det = torch.tensor(det)
                    name = img.split(os.sep)[-2]  # 以文件分割符来进行分割
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    # writer.writerow([img, label, det])
                    writer.writerow([img, label])
                print('write into csv file:', filename)

        # images, labels, landmarks = [], [], []
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # img, label, landmark = row
                img, label = row
                label = int(label)
                # landmark = literal_eval(landmark)

                images.append(img)
                labels.append(label)
                # landmarks.append(landmark)
        assert len(images) == len(labels)  # 保证长度一致

        return images, labels

    def __len__(self):  # 样本数目
        return len(self.images)

    def denormalize(self, x_hat):  # 逆归一化
        '''mean = [0.512, 0.512, 0.512]
        std = [0.257, 0.257, 0.257]'''
        mean = [0.507, 0.507, 0.507]
        std = [0.212, 0.212, 0.212]

        # x_hat = (x-mean)/std
        # x = x_hat*std + mean
        # x:[c: h : w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):  # 索引
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0

        # img, label, landmark = self.images[idx], self.labels[idx], self.landmarks[idx]
        img, label = self.images[idx], self.labels[idx]
        mu, st = 0, 255

        '''tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize([0.507, 0.507, 0.507],
                                    [0.212, 0.212, 0.212])
        ])
            '''
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.RandomApply([transforms.RandomResizedCrop(48, scale=(0.8, 1.2))], p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                 [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.4),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),

            # transforms.FiveCrop(40),  # 将data转变为5D 需要转为4D
            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            # transforms.Lambda(lambda tensors: torch.stack(
            #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop(crops, 0, 0, 40, 40)),
                                                          transforms.ToTensor()(crop(crops, 0, 8, 40, 40)),
                                                          transforms.ToTensor()(crop(crops, 8, 0, 40, 40)),
                                                          transforms.ToTensor()(crop(crops, 4, 4, 40, 40)),
                                                          transforms.ToTensor()(crop(crops, 8, 8, 40, 40))])),
                                                          # transforms.Resize((40, 40))(
                                                          # transforms.ToTensor()(crop(crops, 0, 0, 48, 48)))])),
             # transforms.Lambda(lambda tensors: torch.stack(
             #    [pad(t, padding=(0, 0, 8, 8), padding_mode='constant') for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing(p=0.5)(t) for t in tensors])),   # 随机遮挡
        ])
        '''tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Grayscale(),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            # transforms.FiveCrop(40),  # 将data转变为5D 需要转为4D
            transforms.RandomCrop((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize([0.507],
                                 [0.212]),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.RandomErasing()(t) for t in tensors])),   # 随机遮挡

        ])'''


        '''tf_test = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.ToTensor(),
            transforms.Resize((40, 40)),
            transforms.Normalize([0.507, 0.507, 0.507],
                                [0.212, 0.212, 0.212])
        ])
        transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,)),
            transforms.FiveCrop(40),'''

        tf_test = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            # transforms.FiveCrop(40),  # 将data转变为5D 需要转为4D

            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop(crops, 0, 0, 40, 40)),
            #                                              transforms.ToTensor()(crop(crops, 0, 8, 40, 40)),
            #                                              transforms.ToTensor()(crop(crops, 8, 0, 40, 40)),
            #                                              transforms.ToTensor()(crop(crops, 4, 4, 40, 40)),
            #                                              transforms.ToTensor()(crop(crops, 8, 8, 40, 40)),
            #                                              transforms.Resize((40, 40))(
            #                                              transforms.ToTensor()(crop(crops, 0, 0, 48, 48)))])),

            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop(crops, 0, 0, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 0, 8, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 8, 0, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 4, 4, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 8, 8, 40, 40))])),
                                                         # transforms.Resize((40, 40))(
                                                         #     transforms.ToTensor()(crop(crops, 0, 0, 48, 48)))])),

            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        ])

        tf_test_2 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            # transforms.FiveCrop(40),  # 将data转变为5D 需要转为4D
            # transforms.Lambda(lambda crops: torch.stack(
            #     [transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop(crops, 0, 0, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 0, 8, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 8, 0, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 4, 4, 40, 40)),
                                                         transforms.ToTensor()(crop(crops, 8, 8, 40, 40))])),
                                                         # transforms.Resize((40, 40))(
                                                          #    transforms.ToTensor()(crop(crops, 0, 0, 48, 48)))])),
            transforms.Lambda(lambda tensors: torch.stack(
                [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        ])
        '''tf_test = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Grayscale(),
            transforms.Resize((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize([0.507],
                                [0.212])

            ])'''
        RGB_MEAN = [0.5, 0.5, 0.5]
        RGB_STD = [0.5, 0.5, 0.5]
        tf_test_orig = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((112, 112)),
            # transforms.Grayscale(),
            # transforms.Resize((40, 40)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(mu,), std=(st,)),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),

        ])
        tf_test_orig_2 = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize((48, 48)),
            transforms.Grayscale(),
            # transforms.Resize((40, 40)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,)),
        ])

        tf_orig = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            # transforms.Grayscale(),
            transforms.Resize((112, 112)),
            transforms.RandomApply([transforms.RandomResizedCrop(112, scale=(0.8, 1.2))], p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
            # transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.4),
            # transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
            transforms.RandomApply([transforms.RandomErasing(p=0.5)], p=0.5)

        ])

        if self.model == 'test' or self.model == 'val':
            # img1 = torch.cat([tf_test(img), torch.unsqueeze(tf_test_orig(img), 0)], dim=0)
            #img2 = torch.cat([tf_test_2(img), torch.unsqueeze(tf_test_orig_2(img), 0)], dim=0)
            # img = torch.cat([img1, img2], dim=0)
            # img = torch.cat([tf_test(img), tf_test_2(img)], dim=0)
            # img = torch.cat([torch.unsqueeze(tf_test_orig(img), 0), torch.unsqueeze(tf_test_orig_2(img), 0)], dim=0)
            # img = tf_test(img)
            img = tf_test_orig(img)
        else:
            # img = torch.cat([tf(img), tf_test_2(img)], dim=0)
            # img = tf(img)
            img = tf_orig(img)
        # landmarks = torch.tensor(landmark)
        label = torch.tensor(label)
        return img, label
    def get_labels(self):
        return self.labels

def main():
    '''db = Pokemon1(r'D:\pythonProject3\FER2013\FER2013\Training', 40, 'training')
    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)

    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        # viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
        time.sleep(5)'''

    val_db = Pokemon1(r'D:\pythonProject3\FER2013\FER2013\Training', 40, mode='train')
    val_loader = DataLoader(val_db, batch_size=128, num_workers=1)
    for x, y in val_loader:
        print(x.shape, y.shape)
        print(x.shape, y.shape)


if __name__ == '__main__':
    main()



