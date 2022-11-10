import pandas as pd

result1 = pd.read_csv('submit1.csv')    # xgb交叉验证 0.9565
result2 = pd.read_csv('金融预测.csv')     # ctb 0.9526
result3 = pd.read_csv('submit2.csv')    # ctb 0.94
result4 = pd.read_csv('submit3.csv')    # lgb 0.9
result1['fraud'] = 0.4*result1['fraud']+0.4*result2['fraud']+0.2*result4['fraud']
result1.to_csv('submitx.csv', index=False)
