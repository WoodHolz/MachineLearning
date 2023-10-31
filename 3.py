import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 1024
#定义每一训练批次的大小
EPOCHS = 10
#定义迭代次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir="D:\Pycharm\Py_Projects\DeepLearn\mnist"
image_files = os.listdir(data_dir)

trans=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
])
dataset = ImageFolder(root=data_dir, transform=trans)

# class GrayImageFolder(ImageFolder):
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         image = self.loader(path)
#         image = image.convert("L")  # 转换为灰度图像
#         if self.transform is not None:
#             image = self.transform(image)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return image, target
# dataset = GrayImageFolder(root=data_dir, transform=trans)

train_data, test_data, train_labels, test_labels = train_test_split(dataset,dataset.targets,train_size=0.8,
                                                                          test_size=0.2, random_state=30)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True)

class ConvNet(nn.Module):
#定义卷积神经网络模型
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)
#该卷积神经网络定义了两个卷积层，两个全连接层
# self.conv1 = nn.Conv2d(1, 10, 5)为第一个卷积层，1为输入通道数，10为输出通道数，5为卷积核大小
#self.conv2 = nn.Conv2d(10, 20, 3)为第二个卷积层，10为输入通道数，20为输出通道数，3为卷积核大小
#self.fc1 = nn.Linear(20 * 10 * 10, 500)为第一个全连接层，输入参数为2000，输出参数为500
#self.fc2 = nn.Linear(500, 10)为第二个全连接层,输入参数数量为500，输出为10（10个数字）
    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
#第一层卷积层，输入批次数据，输出为24*24*10
        out = F.relu(out)
#激活函数，负数置为0，整数置为1
        out = F.max_pool2d(out, 2, 2)
#最大池化层，池化窗口大小为2，步长为2，输出为12*12*10
        out = self.conv2(out)
#第二层卷积层，输出为10*10*20
        out = F.relu(out)
#激活函数
        out = out.view(in_size, -1)
#转为一维张量
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
#将输出按行做归一化
        return out

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%40 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        torch.save(model, 'work.pkl')  # 保存整个网络,保存路径为当前路径

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
