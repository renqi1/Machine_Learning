"本质是线性回归+sigmoid，用于二分类，处理多分类任务时有OVR和MVM两种方法"
"线性模型，因为决策边界是线性的"
# 基础函数库
import numpy as np
# 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns
# 导入逻辑回归模型函数
from sklearn.linear_model import LogisticRegression
# Demo演示LogisticRegression分类
# 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])
# 调用逻辑回归模型
lr_clf = LogisticRegression()
# 用逻辑回归模型拟合构造的数据集
lr_clf = lr_clf.fit(x_fearures, y_label)    # 其拟合方程为 y=w0+w1*x1+w2*x2
# 查看其对应模型的w
print('the weight of Logistic Regression:', lr_clf.coef_)
# 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:', lr_clf.intercept_)
# 可视化构造的数据样本点
# 可视化决策边界
plt.figure()
plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
plt.title('Dataset')
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(), y_grid.ravel()])   # 返回的是每个属于每个类别的概率
z_proba = z_proba[:, 1].reshape(x_grid.shape)   # 取第一列，类别为1的概率，重构为200*100
plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')   # 分界线为0.5
plt.show()
# 测试
x_test = np.array([[0, -1], [1, 2]])
print(lr_clf.predict(x_test))
print(lr_clf.predict_proba(x_test))
