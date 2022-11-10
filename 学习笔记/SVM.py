"也是针对二分类，对分类做法和逻辑回归一样"
# 基础函数库
import numpy as np
# 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns
# 导入逻辑回归模型函数
from sklearn import svm

# Demo演示LogisticRegression分类
# 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])
# 调用SVC模型 （支持向量机分类）
svc = svm.SVC(kernel='linear')
# 用SVM模型拟合构造的数据集
svc = svc.fit(x_fearures, y_label)

# 查看其对应模型的w
print('the weight of Logistic Regression:', svc.coef_)
# 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:', svc.intercept_)
y_train_pred = svc.predict(x_fearures)
print('The predction result:',y_train_pred)

# 最佳函数
x_range = np.linspace(-3, 3)
w = svc.coef_[0]
a = -w[0] / w[1]
y_3 = a*x_range - (svc.intercept_[0]) / w[1]    #最佳函数
# 可视化决策边界
plt.figure()
plt.scatter(x_fearures[:, 0], x_fearures[:, 1], c=y_label, s=50, cmap='viridis')
plt.plot(x_range, y_3, '-c')
plt.show()