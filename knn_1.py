from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
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

    # 将多类别标签转换为二分类标签
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_predict_bin = label_binarize(y_predict, classes=[0, 1, 2])

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_predict_bin.ravel())
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_predict_bin.ravel())
    average_precision = average_precision_score(y_test_bin, y_predict_bin, average='micro')

    # 绘制散点图、PR曲线和ROC曲线
    plt.figure(figsize=(15, 5))

    # 散点图
    plt.subplot(1, 3, 1)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_predict)
    # \ , label='Prediction') # 怎么尝试加入名字?????????  (*/ω＼*)
    # plt.legend()

    plt.legend()
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('KNN Iris Classification')



    # 绘制ROC曲线
    plt.subplot(1, 3, 2)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 绘制PR曲线
    plt.subplot(1, 3, 3)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    plt.tight_layout()
    plt.show()

    return None

knn_iris()