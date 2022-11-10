"基础决策树是靠信息增益选择判断条件，后续有许多改进算法如ID3,C4.5,CART,CART靠GINI系数判定，原则是希望纯度越高越好"
"criterion可选entropy,gini"
"max_depth限制深度，min_samples_leaf最小叶子数，小于该参数不往下分"
#  基础函数库
import numpy as np
# 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns
# 导入决策树模型函数
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Demo演示LogisticRegression分类
# 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 1, 0, 1, 0, 1])
# 调用决策树回归模型
tree_clf = DecisionTreeClassifier()
# 调用决策树模型拟合构造的数据集
tree_clf = tree_clf.fit(x_fearures, y_label)

# # 可视化决策树
# import graphviz   # pip instal前需要到官网下载安装并设置环境变量，懒得下了
# dot_data = tree.export_graphviz(tree_clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("pengunis")
# # 参考https://blog.csdn.net/qq_38235178/article/details/108096106

# 创建新样本
x_fearures_new1 = np.array([[0, -1]])
x_fearures_new2 = np.array([[2, 1]])
# 在训练集和测试集上分布利用训练好的模型进行预测
y_label_new1_predict = tree_clf.predict(x_fearures_new1)
y_label_new2_predict = tree_clf.predict(x_fearures_new2)
print('The New point 1 predict class:\n', y_label_new1_predict)
print('The New point 2 predict class:\n', y_label_new2_predict)