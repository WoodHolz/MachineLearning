import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn

BATCH_SIZE = 1024
epochs = 25  # 迭代次数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def data_read():
    data_dir="D:\Pycharm\Py_Projects\DeepLearn\mnist"
    image_files = os.listdir(data_dir)

    def flatten_vector(tensor):
        return tensor.view(-1)

    trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(flatten_vector),
    ])
    dataset = ImageFolder(root=data_dir,transform=trans)

    class GrayImageFolder(ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = self.loader(path)
            image = image.convert("L")  # 转换为灰度图像
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return image, target
    dataset = GrayImageFolder(root=data_dir, transform=trans)
    train_data, test_data, train_labels, test_labels = train_test_split(dataset,dataset.targets,train_size=0.85,
                                                                    test_size=0.15, random_state=30,)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True)
    return train_loader, test_loader
def data_tf(image):
    """
    数据归一化并将维度展开：[28,28]->[1，784]
    :param image:
    :return:
    """
    img = np.array(image, dtype='float32') / 255
    # img = (img - 0.5) / 0.5
    img = img.reshape((-1,))
    img = torch.from_numpy(img)
    return img
class BPNNMdel(torch.nn.Module):
    def __init__(self):
        super(BPNNMdel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(784, 1000), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(1000, 2000), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(2000, 1000), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(1000, 10))  # 输出维度必须大于标签的维度，即最好大于分类数，否则报错

    def forward(self, img):
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img
def train_model(train_data, test_data, iterations, model, model_criterion, model_optimizer):
    """
    模型训练和评估函数，完成模型训练的整个过程
    :param train_data: 训练用数据集
    :param test_data: 测试用数据集
    :param iterations: 训练迭代的次数
    :param model: 神经网络模型
    :param model_criterion: 损失函数
    :param model_optimizer: 反向传播优化函数
    :return:
    """
    model_train_losses = []
    model_train_acces = []
    model_eval_losses = []
    model_eval_acces = []
    for epoch in range(iterations):
        # 网络训练
        train_loss = 0
        train_acc = 0
        model.train()
        for i, data in enumerate(train_data):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = model_criterion(out, label)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
        model_train_losses.append(train_loss / len(train_data))
        model_train_acces.append(train_acc / len(train_data))

        # 网络评估
        eval_loss = 0
        eval_acc = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_data):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                out = model(img)
                loss = model_criterion(out, label)

                eval_loss += loss.item()

                _, pred = out.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / img.shape[0]
                eval_acc += acc
            model_eval_losses.append(eval_loss / len(test_data))
            model_eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(epoch+1, train_loss / len(train_data), train_acc / len(train_data),
                      eval_loss / len(test_data), eval_acc / len(test_data)))
    return model_train_losses, model_train_acces, model_eval_losses, model_eval_acces

if __name__ == "__main__":
    train_load, test_load = data_read()
    learning_rate = 0.01  # 学习率
    Model = BPNNMdel()
    Model = Model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Model.parameters(), lr=learning_rate)
    train_losses, train_acces, eval_losses, eval_acces = train_model(
        train_data=train_load, test_data=test_load, iterations=epochs,
        model=Model, model_criterion=criterion, model_optimizer=optimizer
    )
