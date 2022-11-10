import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv("submission.csv")
data = pd.concat([train, test], axis=0, ignore_index=True)

# 把所有的相同类别的特征编码为同一个值
numerical_features = [x for x in data.columns if data[x].dtype != object]
category_features = [x for x in data.columns if data[x].dtype == object]
def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(), range(len(x.unique().tolist()))))
    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction
for i in category_features:
    data[i] = data[i].apply(get_mapfunction(data[i]))

y_train_ = data["subscribe"][:train.shape[0]]
data = data.drop(["id"], axis=1)
data = data.drop(["subscribe"], axis=1)
x_train_ = data[:train.shape[0]]
x_test_ = data[train.shape[0]:]
x_train = np.array(x_train_)
y_train = np.array(y_train_)
x_test = np.array(x_test_)

'0.9596'
from lightgbm.sklearn import LGBMClassifier
# # 从sklearn库中导入网格调参函数
# from sklearn.model_selection import GridSearchCV
# # 定义参数取值范围, 网格调参，它的基本思想是穷举搜索
# learning_rate = [0.1]
# feature_fraction = [0.5]
# num_leaves = [16]
# max_depth = [-1]
# parameters = {'learning_rate': learning_rate,
#               'feature_fraction':feature_fraction,
#               'num_leaves': num_leaves,
#               'max_depth': max_depth}
# model = LGBMClassifier(n_estimators = 50)
# # 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
# clf = clf.fit(x_train, y_train)
# # 网格搜索后的最好参数为
# print(clf.best_params_)

clf = LGBMClassifier(feature_fraction=0.5, learning_rate=0.1, max_depth=-1, num_leaves=16)
clf.fit(x_train, y_train)
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
result = list(test_predict)
submission['subscribe'] = result
submission['subscribe'] = submission['subscribe'].map(lambda x: 'no' if x == 0 else 'yes')
submission.to_csv("submit3.csv", index=False)
