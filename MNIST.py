import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 1024
#定义每一训练批次的大小
EPOCHS = 10
#定义迭代次数
DEVICE = torch.device = torch.device("cuda")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
#torch.utils.data.DataLoader类主要用于数据读取
#此处定义了train_loader的对象，
#将torch.utils.data.DataLoader的形参dataset设置为MINST类
#形参batch_size设置默认值，若不进行修改，则默认值为1
#形参shuffl设置为True，该参数若设置为True，则批次训练会将所有数据打乱进行训练，设置为False，则不进行乱序排列，默认值为False
#MNIST中，形参root用于MNIST数据集的保存的位置，此处保存在当前目录下的data文件夹下
#形参train用于当前数据集是否使用训练集，True使用训练集，False使用测试集
#形参download用于是否自动下载数据集，默认值为True
#trasnform用于是否数据集进行转换操作
#transforms.Compose类主要是用于多个图片串联一起，其中参数为一个列表，即列表中的元素为想执行的操作
#transforms.ToTensor()将数据转换为张量
#transforms.Normalize将数据进行归一化处理，即将数据均值变为0.1307，标准差（方差的算数平方根）变为0.3081
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)
#此处进改变MNIST参数train，即设置测试集

class ConvNet(nn.Module):
#定义卷积神经网络模型
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
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
#优化器定位随机梯度下降

def train(model, device, train_loader, optimizer, epoch):
    model.train()
#将模型设置为训练模式，能够启用dropout和batch normalization等正则化技术来防止过拟合
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
#梯度置0，反向传播时可以避免梯度积累的影响（这里不是很懂？？）
        output = model(data)
        loss = F.nll_loss(output, target)
#损失函数定位交叉熵
        loss.backward()
#反向传播
        optimizer.step()
#更新模型参数
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} 0[{}/{} ({:.f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            for i in range(len(target)):
                digit = target[i].item()
                save_path = os.path.join(digit_dirs[digit], f"train_image_{i}.png")
                torchvision.utils.save_image(data[i], save_path)
    torch.save(model, 'MinstNet.pkl')  # 保存整个网络,保存路径为当前路径
def test(model, device, test_loader):
    model.eval()
#将模型设置为评估模式
    test_loss = 0
#计算损失
    correct = 0
#正确预测的数量
    save_dir = "classification"
    os.makedirs(save_dir, exist_ok=True)
#创建“classification”的文件夹
    digit_dirs = [save_dir+'\\'+str(i) for i in range(10)]
    for dir in digit_dirs:
        os.makedirs(dir, exist_ok=True)
#创建二级子目录
    with torch.no_grad():
#这个with不是很懂？
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
#计算损失
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            for i in range(len(target)):
                digit = target[i].item()
                save_path = os.path.join(digit_dirs[digit], f"image_{i}.png")
                torchvision.utils.save_image(data[i], save_path)
#没看懂？？

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)