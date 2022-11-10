# 导入乳腺癌数据集
from sklearn.datasets import load_breast_cancer
# 导入BP模型
from sklearn.neural_network import MLPClassifier
# 导入训练集分割方法
from sklearn.model_selection import train_test_split
# 导入预测指标计算函数和混淆矩阵计算函数
from sklearn.metrics import classification_report, confusion_matrix
# 导入绘图包
import seaborn as sns
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
# 查看数据集信息
print('breast_cancer数据集的长度为：', len(cancer))
print('breast_cancer数据集的类型为：', type(cancer))
# 分割数据为训练集和测试集
cancer_data = cancer['data']
print('cancer_data数据维度为：', cancer_data.shape)
cancer_target = cancer['target']
print('cancer_target标签维度为：', cancer_target.shape)
cancer_names = cancer['feature_names']
cancer_desc = cancer['DESCR']
# 分为训练集与测试集
cancer_data_train, cancer_data_test = train_test_split(cancer_data, test_size=0.2, random_state=42)   # 训练集
cancer_target_train, cancer_target_test = train_test_split(cancer_target, test_size=0.2, random_state=42) # 测试集

# 建立 BP 模型, 采用Adam优化器，relu非线性映射函数
BP = MLPClassifier(solver='adam', activation='relu', max_iter=1000, alpha=1e-3, hidden_layer_sizes=(64, 32, 32), random_state=1)
# 进行模型训练
BP.fit(cancer_data_train, cancer_target_train)

# 进行模型预测
# 显示预测分数
print("预测准确率: {:.4f}".format(BP.score(cancer_data_test, cancer_target_test)))
# 进行测试集数据的类别预测
predict_test_labels = BP.predict(cancer_data_test)
print("测试集的真实标签:\n", cancer_target_test)
print("测试集的预测标签:\n", predict_test_labels)
# 进行预测结果指标统计 统计每一类别的预测准确率、召回率、F1分数
print(classification_report(cancer_target_test, predict_test_labels))
# 计算混淆矩阵
confusion_mat = confusion_matrix(cancer_target_test, predict_test_labels)
# 打混淆矩阵
print(confusion_mat)
# 将混淆矩阵以热力图的防线显示
sns.set()
figure, ax = plt.subplots()
# 画热力图
sns.heatmap(confusion_mat, cmap="YlGnBu_r", annot=True, ax=ax)
# 标题
ax.set_title('confusion matrix')
# x轴为预测类别
ax.set_xlabel('predict')
# y轴实际类别
ax.set_ylabel('true')
plt.show()
