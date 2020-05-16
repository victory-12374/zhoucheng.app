from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import joblib
train_data = pd.read_csv('F:/Last_Data/Last_Data2/last_test.csv')#经过特征提取后的训练集
train_data_y = train_data['label']
train_data_x = train_data.drop(['label'],axis=1)

model_gbdt_default = GradientBoostingClassifier(random_state=500)
#模型训练
model_gbdt_default.fit(train_data_x,train_data_y)

joblib.dump(model_gbdt_default,'F:/Last_Python/code1/GBDT_model9.model')
