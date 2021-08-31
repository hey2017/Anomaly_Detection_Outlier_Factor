
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

os.chdir(directory)


df2 = pd.read_csv('Housing_Data.csv')



data = df2.values


x, y = data[:,:-1], data[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 1)



contamination_list = list(np.linspace(0.001,0.5,num=100))

#%%
result_lst =[]

for i in contamination_list:
    x_train2 = 0
    y_train2 = 0
    y_pred_linear = 0
    y_pred2_linear= 0
    
    model = LocalOutlierFactor (contamination = i)
    y_pred = model.fit_predict(x_train)
    tag = y_pred != -1
    x_train2, y_train2 = x_train[tag,:], y_train[tag]
    
    linear_model = LinearRegression()
    #before detecting the outliers
    linear_model.fit(x_train, y_train)
    y_pred_linear = linear_model.predict(x_test)
    #after detecting the outliers
    linear_model.fit(x_train2, y_train2)
    y_pred2_linear = linear_model.predict(x_test)
    
    result_lst.append((i, mean_absolute_error(y_test,y_pred_linear), mean_absolute_error(y_test,y_pred2_linear)))
#%%
result = pd.DataFrame(result_lst)
result.columns = ['contamination value', 'mae before fit', 'mae after fit']
result = result.sort_values(by='mae after fit', ascending = True)

print('The best contamination value is:\n ', result.iloc[0])

#%%
mae_after = []
contamination_val = []

for i in range(len(result_lst)):
    mae_after.append(result_lst[i][2])
    contamination_val.append(result_lst[i][0])
    
plt.plot(contamination_val, mae_after,'--*')
plt.ylabel('absolute mean square after fit')
plt.xlabel('contamination value')
