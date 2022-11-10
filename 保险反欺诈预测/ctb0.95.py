# 参考：https://blog.csdn.net/qq_44694861/article/details/125572858?spm=1001.2014.3001.5501

import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
# warnings.filterwarnings('ignore')
# %matplotlib inline
from sklearn.metrics import roc_auc_score
# 数据降维处理的
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv("submission.csv")
data = pd.concat([train,test])
data['incident_date'] = pd.to_datetime(data['incident_date'],format='%Y-%m-%d')

startdate = datetime.datetime.strptime('2022-06-30', '%Y-%m-%d')
data['time'] = data['incident_date'].apply(lambda x: startdate-x).dt.days

numerical_fea = list(data.select_dtypes(include=['object']).columns)
division_le = LabelEncoder()
for fea in numerical_fea:
    division_le.fit(data[fea].values)
    data[fea] = division_le.transform(data[fea].values)
print("数据预处理完成!")

testA = data[data['fraud'].isnull()].drop(['policy_id','incident_date','fraud'],axis=1)
trainA = data[data['fraud'].notnull()]
data_x = trainA.drop(['policy_id','incident_date','fraud'],axis=1)
data_y = train[['fraud']].copy()
col = ['policy_state','insured_sex','insured_education_level','incident_type','collision_type','incident_severity','authorities_contacted','incident_state',
     'incident_city','police_report_available','auto_make','auto_model']
for i in data_x.columns:
    if i in col:
        data_x[i] = data_x[i].astype('str')
for i in testA.columns:
    if i in col:
        testA[i] = testA[i].astype('str')
model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            learning_rate=0.1,
            iterations=10000,
            random_seed=2020,
            od_type="Iter",
            depth=7,
            early_stopping_rounds=300)
answers = []
mean_score = 0
n_folds = 10
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)
for train, test in sk.split(data_x, data_y):
    x_train = data_x.iloc[train]
    y_train = data_y.iloc[train]
    x_test = data_x.iloc[test]
    y_test = data_y.iloc[test]
    clf = model.fit(x_train,y_train, eval_set=(x_test,y_test), verbose=500, cat_features=col)
    yy_pred_valid = clf.predict(x_test)
    print('cat验证的auc:{}'.format(roc_auc_score(y_test, yy_pred_valid)))
    mean_score += roc_auc_score(y_test, yy_pred_valid) / n_folds
    y_pred_valid = clf.predict(testA, prediction_type='Probability')[:, -1]
    answers.append(y_pred_valid)
print('10折平均Auc:{}'.format(mean_score))
lgb_pre = sum(answers)/n_folds
sub['fraud'] = lgb_pre
sub.to_csv('金融预测.csv', index=False)