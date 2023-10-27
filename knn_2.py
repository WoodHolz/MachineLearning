class KNNClassify():

    def __init__(self,k=8, p=2):
        self.k = k
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict_y(self, X_test):
        m = self._X_train.shape[0]
        y_pre = []
        for intX in X_test:
            minus_mat = np.fabs(np.tile(intX, (m, 1)) - self._X_train)       # 将新的实例复制成m行1列，并进行相减
            sq_minus_mat = minus_mat ** self.p
            sq_distance = sq_minus_mat.sum(axis=1)
            diff_sq_distance = sq_distance ** float(1/self.p)

            sorted_distance_index = diff_sq_distance.argsort()               # 记录距离最近的k个点的索引
            class_count = {}
            vola = []
            for i in range(self.k):
                vola = self._y_train[sorted_distance_index[i]]
                class_count[vola] = class_count.get(vola, 0) + 1             # 统计k个点中所属各个类别的实例数目

            sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)    # 返回列表，元素为元组。每个类别以及对应的实例数目
            y_pre.append((sorted_class_count[0][0]))
        return (np.array(y_pre))

    def score(self, X_test, y_test):
        j = 0
        for i in range(len(self.predict_y(X_test))):
            if self.predict_y(X_test)[i] == y_test[i]:
                j += 1
        return ('accuracy: {:.10%}'.format(j / len(y_test)))

import numpy as np
import operator

from sklearn import datasets
from sklearn.model_selection import train_test_split
# 获取数据集，并进行8:2切分
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 定义分类器的实例，并进行拟合预测
f = KNNClassify()
f.fit(X_train, y_train)
y_pre = f.predict_y(X_test)
accuracy = f.score(X_test, y_test)
print(y_test)
print(y_pre)
print(accuracy)