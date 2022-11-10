# Demo来自sklearn官网
"回归就是找附近K个点的平均值，k为1预测结果等于最近值，k为N结果就是平均值了，感觉这种拟合效果不好"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(0)
# 随机生成40个(0, 1)之前的数，乘以5，再进行升序
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# 创建[0, 5]之间的500个数的等差数列, 作为测试数据
T = np.linspace(0, 5, 500)[:, np.newaxis]
# 使用sin函数得到y值，并拉伸到一维
y = np.sin(X).ravel()
# Add noise to targets[y值增加噪声]
y[::5] += 1 * (0.5 - np.random.rand(8))    # 每隔五个数加入噪声
# #############################################################################
# Fit regression model
# 设置多个k近邻进行比较
n_neighbors = [1, 3, 5, 8, 40]
# 设置图片大小
plt.figure(figsize=(10,20))
for i, k in enumerate(n_neighbors):
    # 默认使用加权平均进行计算predictor
    clf = KNeighborsRegressor(n_neighbors=k, p=2, metric="minkowski")
    # 训练
    clf.fit(X, y)
    # 预测
    y_ = clf.predict(T)
    plt.subplot(6, 1, i + 1)
    plt.scatter(X, y, color='red', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i)" % (k))

plt.tight_layout()
plt.show()

# k太小过拟合，太大欠拟合