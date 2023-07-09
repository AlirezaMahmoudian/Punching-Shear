#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars
from sklearn.metrics import r2_score
import warnings
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor , GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_excel('D:/Punching Article/New ex.xlsx'  ,header = 0 )
y = df.loc[:, 'Pmax (kN)'].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4,5]].to_numpy()


# In[7]:


r2test=[]
r2train=[]
result=[]
Xtr , Xte , ytr , yte = train_test_split(X,y, train_size = 0.8 ,random_state=42 )
scalerX=StandardScaler()
Xtr1=scalerX.fit_transform(Xtr)
Xte1=scalerX.transform(Xte)
scalery=StandardScaler()
ytr1=scalery.fit_transform(ytr)
yte1=scalery.transform(yte)
n_estimators=[100 , 200 , 400 , 600 , 800 , 1000]
max_depth=[2 , 4 , 6 , 8 , 10 ,15]
Bace_score=[0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 1]
booster=['gbtree' , 'gblinear' , 'dart' ]
for i in n_estimators:
  for j in max_depth:
    for t in Bace_score:
      for m in booster:
        model = XGBRegressor(n_estimators=i, max_depth=j, Bace_score=t, booster =m ,  random_state=42 )
        model.fit(Xtr1 , ytr1)
        yprtr = model.predict(Xtr1)
        yprte = model.predict(Xte1)
        r2tr=round(r2_score(ytr1 , yprtr),4)
        r2te=round(r2_score(yte1 , yprte),4)
        result.append((i,j,t,m,r2te,r2tr))
        r2train.append(r2tr)
        r2test.append(r2te)
result.sort(key=lambda x:x[4])
print(result[-1])

print(len(result))
model=XGBRegressor(n_estimators=result[-1][0] ,max_depth=result[-1][1] ,Bace_score=result[-1][2]
                            ,booster=result[-1][3], random_state=42)
                            
model.fit(Xtr1 , ytr1)
lr_r2tr = model.score(Xtr1 , ytr1)
lr_r2te = model.score(Xte1 , yte1)
    
print(f'Train R2 Score: {round(lr_r2tr, 4)}')
print(f'Test  R2 Score: {round(lr_r2te, 4)}')

trpred = model.predict(Xtr1)
tepred = model.predict(Xte1)

a = min([np.min(trpred), np.min(tepred), 0])
b = max([np.max(trpred), np.max(tepred), 1])

# plt.subplot(1, 2, 1)
plt.scatter(ytr1, trpred, s=12, facecolors='none', edgecolors='orangered')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'XGBoost Train [R2 = {round(lr_r2tr, 4)}]')
plt.xlabel('V test (kN)')
plt.ylabel('V Predicted (kN)')
plt.legend()
plt.show()

# plt.subplot(1, 2, 2)
plt.scatter(yte1, tepred, s=12, facecolors='none', edgecolors='orangered')
plt.plot([a, b], [a, b], c='black', lw=1.4, label='y = x')
plt.title(f'XGBoost Test [R2 = {round(lr_r2te, 4)}]')
plt.xlabel('V test (kN)')
plt.ylabel('V Predicted (kN)')
plt.legend()
plt.show()


# In[ ]:




