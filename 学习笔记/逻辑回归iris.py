#  基础函数库
import numpy as np
import pandas as pd
# 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
# 我们利用 sklearn 中自带的 iris 数据作为数据载入，并利用Pandas转化为DataFrame格式
from sklearn.datasets import load_iris

data = load_iris()  # 得到数据特征
iris_target = data.target   # 得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)    # 利用Pandas转化为DataFrame格式
# 利用.info()查看数据的整体信息
iris_features.info()
# 进行简单的数据查看，我们可以利用 .head() 头部.tail()尾部
print(iris_features.head())
# 合并标签和特征信息
iris_all = iris_features.copy() # 进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target
print(iris_all.head())

'特征与标签组合的散点可视化'
# sns.pairplot(data=iris_all, diag_kind='hist', hue= 'target')     #绘制两两之间的关系，针对'target'进行分类，diag_kind对角线为hist
# plt.show()

'箱型图'
# for col in iris_features.columns:
#     sns.boxplot(x='target', y=col, saturation=0.5,palette='pastel', data=iris_all)  #x,y为列名，data为dataframe或数组
#     plt.title(col)
#     plt.show()

'选取其前三个特征绘制三维散点图'
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')
# iris_all_class0 = iris_all[iris_all['target']==0].values
# iris_all_class1 = iris_all[iris_all['target']==1].values
# iris_all_class2 = iris_all[iris_all['target']==2].values
# # 'setosa'(0), 'versicolor'(1), 'virginica'(2)
# ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
# ax.scatter(iris_all_class1[:,0], iris_all_class1[:,1], iris_all_class1[:,2],label='versicolor')
# ax.scatter(iris_all_class2[:,0], iris_all_class2[:,1], iris_all_class2[:,2],label='virginica')
# plt.legend()
# plt.show()

'二分类'
# # 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
# from sklearn.model_selection import train_test_split
# # 选择其类别为0和1的样本 （不包括类别为2的样本）
# iris_features_part = iris_features.iloc[:100]
# iris_target_part = iris_target[:100]
# # 测试集大小为20%， 80%/20%分
# x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size=0.2, random_state=2020)
# # 从sklearn中导入逻辑回归模型
# from sklearn.linear_model import LogisticRegression
# # 定义 逻辑回归模型
# clf = LogisticRegression(random_state=0, solver='lbfgs')
# clf.fit(x_train, y_train)
# # 查看其对应的w
# print('the weight of Logistic Regression:' clf.coef_)
# # 查看其对应的w0
# print('the intercept(w0) of Logistic Regression:', clf.intercept_)
# ## 在训练集和测试集上分布利用训练好的模型进行预测
# train_predict = clf.predict(x_train)
# test_predict = clf.predict(x_test)
# from sklearn import metrics
# # 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
# print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
# print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
# # 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
# confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
# print('The confusion matrix result:\n', confusion_matrix_result)
# # 利用热力图对于结果进行可视化
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.show()

'三分类'
# 测试集大小为20%， 80%/20%分
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(iris_features, iris_target, test_size=0.2, random_state=2020)
clf = LogisticRegression(random_state=0, solver='lbfgs')
clf.fit(x_train, y_train)
# 查看其对应的w
print('the weight of Logistic Regression:\n', clf.coef_)
# 查看其对应的w0
print('the intercept(w0) of Logistic Regression:\n', clf.intercept_)
# 由于这个是3分类，所有我们这里得到了三个逻辑回归模型的参数，其三个逻辑回归组合起来即可实现三分类。
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
# 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用 predict_proba 函数预测其概率
train_predict_proba = clf.predict_proba(x_train)
test_predict_proba = clf.predict_proba(x_test)
print('The test predict Probability of each class:\n', test_predict_proba)
# 其中第一列代表预测为0类的概率，第二列代表预测为1类的概率，第三列代表预测为2类的概率。
# 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
# 查看混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)
# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()