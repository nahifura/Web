import torch.nn.functional as F
from model.main import *

import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

if __name__ == '__main__':
    set_seed(42)

if __name__ == '__main__':
    # 定义常数
    train_path = os.path.join('.', 'train')
    test_path = os.path.join('.', 'test')
    learning_rate = 0.001
    epochs = 32
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义继承了dataset的类


class FlowerDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.images_path = []
        self.labels = []

        classes = sorted(os.listdir(self.root_dir))
        for cls_name in classes:
            cls_dir = os.path.join(root_dir,cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir,img_name)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                        self.images_path.append(img_path)
                        self.labels.append(classes.index(cls_name))
                except:
                    print(f'skip wrong:{img_path}')
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    # 读取数据转化为dataset
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.ColorJitter(brightness=0.1, contrast=0.01, saturation=0.1,
                                                                hue=0.02)])
    train_dataset = FlowerDataset(root_dir=train_path, transform=train_transform)

    test_dataset = FlowerDataset(root_dir=test_path, transform=test_transform)
    vis_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])
    print(len(train_dataset))
    print(len(test_dataset))


# 展示处理后的图片
def visualize_augmentation(dataset, index=0):
    img_path = dataset.images_path[index]
    original_img = Image.open(img_path).convert('RGB')

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')


    for i in range(4):
        augmented_img = vis_transform(original_img)

        img_show = augmented_img.permute(1, 2, 0).numpy()
        axes[i + 1].imshow(img_show)
        axes[i + 1].set_title(f"Augmented {i + 1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    for o in range(2):
        visualize_augmentation(test_dataset, index=o)

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, persistent_workers=True)

import matplotlib.pyplot as plt
from IPython import display
import time


def use_svg_display():
    display.set_matplotlib_formats('svg')



def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# Animator类（仿照d2l）

class Animator:

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):

        if legend is None:
            legend = []
        use_svg_display()

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):

        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla() #
        for x_data, y_data, fmt in zip(
                self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)
        self.axes[0].grid()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

# 按照d2l写个计时器

class Timer:
    def __init__(self):
        self.times = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def sum(self):
        return sum(self.times)

# 按照d2l写个累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#计算准确率
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())



# 改变的地方：定义了残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super(MyResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




# 准确率测试函数

def evaluate_accuracy_gpu(net,dataloader,device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
        metrix = Accumulator(2)
        with torch.no_grad():
            for X, y in dataloader:
                if isinstance(X, list):
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metrix.add(accuracy(net(X), y), y.numel())
        return metrix[0] / metrix[1]

if __name__ == '__main__':
    net = MyResNet18()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, )



# 定义训练函数
def train(net,train_loader,test_loader,
          epoches,lr,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('res training on',device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()
    timer,num_batches = Timer(), len(train_loader)

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(epoches):
        metrix = Accumulator(3)
        net.train()
        timer.start()
        for i,(X,y) in enumerate(train_loader):
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metrix.add(l.item() * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss = metrix[0]/metrix[2]
            train_accuaracy = metrix[1]/metrix[2]

        print(timer.stop())
        test_acc = evaluate_accuracy_gpu(net,test_loader)
        test_accs.append(test_acc)
        train_losses.append(train_loss)
        train_accs.append(train_accuaracy)
        print('epoch : ',epoch,'finished')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', marker='s')
    plt.plot(range(1, epochs + 1), test_accs, label='Test Acc', marker='^')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)

    # 保存图片到本地
    plt.savefig('res_training_curve.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 res_training_curve.png")

    plt.show()

    print(f'loss:{train_loss:.3f},train_accuaracy:{train_accuaracy:.3f},'
              f'test_acc:{test_acc:.3f}')
    print(f'{metrix[2] * epoches / timer.sum():.3f} examples/sec'f'on {str(device)}')
if __name__ == '__main__':
    train(net, train_loader, test_loader, epochs, learning_rate, device)
    torch.save(net.state_dict(), 'res_best_model.pth')







