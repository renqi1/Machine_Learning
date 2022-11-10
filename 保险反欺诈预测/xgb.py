import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv("submission.csv")
data = pd.concat([train, test], axis=0, ignore_index=True)


# 处理时序数据
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'], format='%Y-%m-%d')
data['incident_date'] = pd.to_datetime(data['incident_date'], format='%Y-%m-%d')
data["year"] = data["policy_bind_date"].dt.year
data["month"] = data["policy_bind_date"].dt.month
data["year1"] = data["incident_date"].dt.year
data["month1"] = data["incident_date"].dt.month
startdate = datetime.datetime.strptime('2022-06-30', '%Y-%m-%d')
data['time'] = data['incident_date'].apply(lambda x: startdate-x).dt.days
data = data.drop(["incident_date"], axis=1)
data = data.drop(["policy_bind_date"], axis=1)

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

y_train_ = data["fraud"][:train.shape[0]]
data = data.drop(["policy_id"], axis=1)
data = data.drop(["fraud"], axis=1)

#归一化
data = (data-data.mean())/data.std()

x_train_ = data[:train.shape[0]]
x_test_ = data[train.shape[0]:]
x_train = np.array(x_train_)
y_train = np.array(y_train_)
x_test = np.array(x_test_)

from xgboost.sklearn import XGBClassifier

# from sklearn.model_selection import GridSearchCV
# learning_rate = [0.01,0.1]
# subsample = [1]
# colsample_bytree = [0.9]
# max_depth = [3,5,7]
# parameters = { 'learning_rate': learning_rate,
#               'subsample': subsample,
#               'colsample_bytree':colsample_bytree,
#               'max_depth': max_depth}
# model = XGBClassifier(n_estimators = 50)
# ## 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=10, scoring='accuracy',verbose=1,n_jobs=-1)
# clf = clf.fit(x_train, y_train)
# print('best params:',clf.best_params_)

# 定义 XGBoost模型
clf = XGBClassifier(colsample_bytree=0.9, learning_rate=0.01, max_depth=5, subsample=1)   # 搜索后最优的
from sklearn.model_selection import KFold
folds = KFold(n_splits=10, shuffle=True, random_state=2022)  # 10折交叉验证
oof_xgb = np.zeros(len(train))  # 用于存放训练集的预测
predictions_xgb = np.zeros(len(test))   # 用于存放测试集的预测
for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    clf.fit(x_train[trn_idx], y_train[trn_idx])
    oof_xgb[val_idx] = clf.predict(x_train[val_idx])  # 预测验证集
    predictions_xgb += clf.predict(x_test) / folds.n_splits  # 预测测试集，并且取平均
submission['fraud'] = predictions_xgb
submission.to_csv('submit1.csv', index=False)

from sklearn.metrics import accuracy_score
print('acc=', accuracy_score(oof_xgb, y_train))

# 这道题用交叉验证就有效果，上一道就没效果，感觉是玄学啊