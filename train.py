import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
from time import time
from skimage.io import imread
import copy
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from torchvision.models import vgg19
from torchsummary import summary

#定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
#每个物体对应的RGB值
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
#每个类的RGB
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128],
                         [0, 128, 128],[128, 128, 128], [64, 0, 0],
                         [192, 0, 0], [64, 128, 0],
                         [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                         [192, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                         [0, 64, 128]]
def image2label(image, colormap):
    #标签转化成每个像素为一类值
    cm2lab = np.zeros(256**3)
    for i, cm in enumerate(colormap):
        cm2lab[(cm[0]*256+cm[1]*256+cm[2])] = i
    image = np.array(image, dtype="int64")
    ix = (image[:, :, 0]*256+image[:, :, 1]*256+image[:, :, 2])
    image2 = cm2lab[ix]
    return image2
#随机裁剪图像数据
def rand_crop(data, label, high, width):
    im_width, im_high = data.size
    #生成图像随机点的位置、
    left = np.random.randint(0, im_width-width)
    top = np.random.randint(0, im_high-high)
    right = left+width
    bottom = top+high
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    return data,label
#单组图像转换操作(数据随机裁剪、 标准化、 二维标签化)
def img_transforms(data, label, high, width, colormap):
    data,label = rand_crop(data, label, high, width)
    data_tfs = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.156, 0.406],
                                                        [0.229, 0.224, 0.225])])
    data = data_tfs(data)
    label = torch.from_numpy(image2label(label, colormap))
    return data, label
def read_image_path(root = "VOC2012/ImageSets/Segmentation/train.txt"):
    image = np.loadtxt(root, dtype=str)
    n = len(image)
    print(n)
    data, label = [None]*n, [None]*n
    for i, fname in enumerate(image):
        data[i] = "VOC2012/JPEGImages/%s.jpg" %(fname)
        label[i] = "VOC2012/SegmentationClass/%s.png" %(fname)
    return data, label
class MyDataset(Data.Dataset):
    def __init__(self, data_root, high, width, imtransform, colormap):
        self.data_root = data_root
        self.high = high
        self.width = width
        self.imtransform = imtransform
        self.colormap = colormap
        data_list, label_list = read_image_path(root=data_root)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[1] >
                                        high and Image.open(im).size[0] >
                                        width)]
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.imtransform(img, label, self.high,
                                       self.width, self.colormap)
        return img, label
    def __len__(self):
        return len(self.data_list)
##读取数据
high, width = 320,480
voc_train = MyDataset("VOC2012/ImageSets/Segmentation/train.txt",
                      high, width, img_transforms, colormap)
voc_val = MyDataset("VOC2012/ImageSets/Segmentation/val.txt", high,
                    width, img_transforms, colormap)
train_loader = Data.DataLoader(voc_train, batch_size=4, shuffle=True,
                               num_workers=8, pin_memory=True)
val_loader = Data.DataLoader(voc_val, batch_size=4, shuffle=True,
                             num_workers=8, pin_memory=True)
##检查训练集的每一个batch的样本维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
print("b_x.shape:", b_x.shape)
print("b_y.shape:", b_y.shape)
#标准话的图像转化为0--1区间
def inv_normalize_iamge(data):
    rgb_mean = np.array([0.458, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0,1)
def label2image(prelabel,colormap):
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w, 3), dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)
        image[index, :] = colormap[ii]
    return image.reshape(h,w,3)

b_x_numpy = b_x.data.numpy()
b_x_numpy = b_x_numpy.transpose(0,2,3,1)
b_y_numpy = b_y.data.numpy()
plt.figure(figsize=(16, 6))
for ii in range(4):
    plt.subplot(2,4,ii+1)
    plt.imshow(inv_normalize_iamge(b_x_numpy[ii]))
    plt.axis("off")
    plt.subplot(2, 4, ii+5)
    plt.imshow(label2image(b_y_numpy[ii],colormap))
    plt.axis("off")
plt.subplots_adjust(wspace=0.1, hspace=0.1)
#plt.show()
model_vgg19 = vgg19(pretrained=True)
base_modle = model_vgg19.features
base_modle.cuda()
summary(base_modle,input_size=(3, high, width))
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes= num_classes
        model_vgg19 = vgg19(pretrained=True)
        self.base_model = model_vgg19.features
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1,
                                          dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512,256,3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128,64,3,2,1,1,1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32,3,2,1,1,1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.layers = {"4": "maxpool_1","9":"maxpool_2",
                       "18":"maxpool_3", "27":"maxpool_4",
                       "36":"maxpool_5"}
    def forward(self, x):
        output = {}
        for name, layer in self.base_model._modules.items():
            x = layer(x)
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output["maxpool_5"]
        x4 = output["maxpool_4"]
        x3 = output["maxpool_3"]
        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score
fcn8s = FCN8s(21).to(device)
summary(fcn8s, input_size=(3, high, width))
def train_model(model, criterion, optimizer, traindataloader, valdataloader, num_cpoches=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_cpoches):
        print('Epoch{}/{}'.format(epoch, num_cpoches-1))
        print('-'*10)
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        model.train()
        for step, (b_x, b_y) in enumerate(traindataloader):
            optimizer.zero_grad()
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            train_loss +=loss.item() * len(b_y)
            train_num += len(b_y)
        train_loss_all.append(train_loss / train_num)
        print('{} Train Loss: {:.4f}'.format(epoch, train_loss_all[-1]))
        model.eval()
        for step,(b_x, b_y) in enumerate(valdataloader):
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)
            out = model(b_x)
            out = F.log_softmax(out, dim=1)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)
        val_loss_all.append(val_loss / val_num)
        print('{} Val Loss: {:.4f}'.format(epoch, val_loss_all[-1]))

        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and Val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    train_process = pd.DataFrame(data={"epoch":range(num_cpoches),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all})
    model.load_state_dict(best_model_wts)
    return model, train_process
LR = 0.0003
criterion = nn.NLLLoss()
optimizer = optim.Adam(fcn8s.parameters(), lr=LR, weight_decay=1e-4)
fcn8s, train_process = train_model(fcn8s, criterion, optimizer, train_loader, val_loader, num_cpoches=30)
torch.save(fcn8s, "fcn8s2.pkl")


#可视化
plt.figure(figsize=(10, 6))
plt.plot(train_process.epoch, train_process.train_loss_all,
         "ro-", label = "Train loss")
plt.plot(train_process.epoch, train_process.val_loss_all,
         "bs-", label = "Val Loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
