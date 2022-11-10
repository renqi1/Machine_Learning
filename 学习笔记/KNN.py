import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
# 使用莺尾花数据集的前两维数据，便于数据可视化
iris = datasets.load_iris()
X = iris.data[:, :2]    # 只取两个特征
y = iris.target

k_list = [1, 3, 5, 8, 10, 15]
h = .02
# 创建不同颜色的画布
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
plt.figure(figsize=(15, 14))
# 根据不同的k值进行可视化
for ind,k in enumerate(k_list):
    clf = KNeighborsClassifier(k)
    clf.fit(X, y)
    # 画出决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 根据边界填充颜色
    Z = Z.reshape(xx.shape)
    plt.subplot(321+ind)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)  # 绘制分类图
    # 数据点可视化到画布
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"% k)
plt.show()