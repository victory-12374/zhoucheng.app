from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

import json
import traceback
import joblib


class Result:
    precision_score=0
    recall_score=0
    accuracy_score=0
    f1_score=0
    roc_auc_score=0

path='F:/Last_Test/test3'#特征提取后的数据集
dir_test=os.listdir(path)
dir_test.sort(key =lambda x:int(x[4:-4]))
input_file=[os.path.join(r'F:/Last_Test/test3',name) for name in dir_test]#特征提取后的数据集
output_file=[os.path.join(r'F:/Last_Test/test5',name) for name in dir_test]#故障诊断之后数据集的存放位置
for x in range(0,len(input_file)):
    params = {}
    params['model'] = 'F:/Last_Python/code1/GBDT_model9.model'
    params['test'] = input_file[x]
    params['opath'] = output_file[x]
    argvs = sys.argv
    try:
        for i in range(len(argvs)):
            if i < 1:
                continue
            if argvs[i].split('=')[1] == 'None':
                params[argvs[i].split('=')[0]] = None
            else:
                Type = type(params[argvs[i].split('=')[0]])
                params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

        model = joblib.load(params['model'])
        
        test1 = pd.DataFrame(pd.read_csv(params['test']))
        
        test_x = test1.drop(['label'],axis=1)
        y_pred = model.predict_proba(test_x)
        
        y_pred = model.predict(test_x)

        predict=np.array(y_pred)
        test=['TEST'+str(x+1)]*len(predict)

        predict=list(predict)
        file_data=pd.DataFrame(predict,columns=['label'])
        file_test=pd.DataFrame(test,columns=['filename'])
        file_all=file_data.join(file_test,how='outer')
        file_all.to_csv(output_file[x],index=False)
    except Exception as e:
        traceback.print_exc()
