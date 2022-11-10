import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from xgboost.sklearn import XGBClassifier

# from sklearn.model_selection import GridSearchCV
# parameters = {'learning_rate': [0.1],
#               'subsample': [1],
#               'colsample_bytree':[0.8],
#               'max_depth': [5]}
# model = XGBClassifier(n_estimators = 50)
# # 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
# clf = clf.fit(x_train, y_train)
# print('best params:', clf.best_params_)

'0.9608'
# 定义 XGBoost模型
clf = XGBClassifier(colsample_bytree=0.8, learning_rate=0.1, max_depth=5, subsample=1)   # 经搜索最优的
# 在训练集上训练XGBoost模型
clf.fit(x_train, y_train)
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
result = list(test_predict)
submission['subscribe'] = result
submission['subscribe'] = submission['subscribe'].map(lambda x: 'no' if x == 0 else 'yes')
submission.to_csv("submit1.csv", index=False)

# feature_importances_去查看特征的重要度。
import seaborn as sns
data_features_part = data[[x for x in data.columns]]
sns.barplot(y=data_features_part.columns, x=clf.feature_importances_)
plt.show()

from sklearn.metrics import accuracy_score
print('acc=', accuracy_score(train_predict, y_train))
