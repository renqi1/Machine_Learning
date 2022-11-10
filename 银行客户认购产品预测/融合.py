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

# 定义 XGBoost模型
from xgboost.sklearn import XGBClassifier
clf1 = XGBClassifier(colsample_bytree=0.8, learning_rate=0.1, max_depth=5, subsample=1)   # 最优的
# 在训练集上训练XGBoost模型
clf1.fit(x_train, y_train)
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict1 = clf1.predict(x_train)
test_predict1 = clf1.predict(x_test)
result_xgb = list(test_predict1)

from lightgbm.sklearn import LGBMClassifier
clf2 = LGBMClassifier(feature_fraction=0.5, learning_rate=0.1, max_depth=-1, num_leaves=16)
clf2.fit(x_train, y_train)
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict2 = clf2.predict(x_train)
test_predict2 = clf2.predict(x_test)
result_lgb = list(test_predict2)

from catboost import CatBoostClassifier
clf3 = CatBoostClassifier(depth=5, learning_rate=0.05, iterations=250)
clf3.fit(x_train, y_train)
# 在训练集和测试集上分布利用训练好的模型进行预测
train_predict3 = clf3.predict(x_train)
test_predict3 = clf3.predict(x_test)
result_ctb = list(test_predict3)

from sklearn.metrics import accuracy_score
print('acc_xgb=', accuracy_score(train_predict1, y_train))
print('acc_lgb=', accuracy_score(train_predict2, y_train))
print('acc_ctb=', accuracy_score(train_predict3, y_train))

result_all = list(test_predict1+test_predict2+test_predict3)
submission['subscribe'] = result_all
submission['subscribe'] = submission['subscribe'].map(lambda x: 'no' if x < 1.5 else 'yes')
submission.to_csv("submit4.csv", index=False)