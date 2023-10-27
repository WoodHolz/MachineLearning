from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def knn_iris():
    # 1）获取数据
    iris = load_iris()
    # 2）划分数据集,训练测试数据
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=8)
    # 3）特征工程：标准化
    transfer = StandardScaler()
    # 实例化一个转换器
    # 对训练集求平均值和标准差，然后进行标准化
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4）KNN算法预估器，k任意取值，这个尝试取6
    estimator = KNeighborsClassifier(n_neighbors=6)
    estimator.fit(x_train, y_train)
    # 拟合分类器
    # 5）模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 可视化
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predict)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('KNN Iris Classification')
    plt.show()

    return None


knn_iris()