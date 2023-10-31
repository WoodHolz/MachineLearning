import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()
x = iris.data
y = iris.target

# 转换数据为Tensor并创建数据集对象
dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义模型
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 30)
        self.fc2 = nn.Linear(30, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 3)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 定义模型和优化器
model = IrisClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# 训练模型
model.train()
for epoch in range(20):
    for x, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, loss: {loss.item():.4f}')

# 在测试数据上评估模型
# 获取模型在测试数据上的预测概率
model.eval()
probabilities = []
true_labels = []
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        probabilities.extend(torch.softmax(y_pred, dim=1).tolist())
        true_labels.extend(y.tolist())

accuracy = 100 * correct / total
print(f'Test accuracy: {accuracy:.2f}%')

# 将预测概率和真实标签转换为NumPy数组
probabilities = np.array(probabilities)
true_labels = np.array(true_labels)

# 计算每个类别的精确度和召回率
precision = dict()
recall = dict()
for i in range(len(iris.target_names)):
    binary_true_labels = np.where(true_labels == i, 1, 0)
    precision[i], recall[i], _ = precision_recall_curve(binary_true_labels, probabilities[:, i])

# 绘制PR曲线
plt.figure()
plt.step(recall[0], precision[0], where='post', label='class 0')
plt.step(recall[1], precision[1], where='post', label='class 1')
plt.step(recall[2], precision[2], where='post', label='class 2')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()