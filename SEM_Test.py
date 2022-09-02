import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random
#random.seed(200)

data = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/Yearly_Data_Normalized/FilteredNormalized2007.csv")
data2 = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/Yearly_Data_Normalized/FilteredNormalized2006.csv")
#data3 = data[data.columns[1:]]
print(data2.head())
#causes = pd.read_csv("/home/local/ASUAD/psheth5/Downloads/CausesofTargetNode2.csv")
causes = pd.read_csv("CausesofTargetNode2.csv")
x = causes.groupby("to")
df = []
for i in x.get_group("-95.35  28.85")['from'].values:
   for j in data.columns:
       temp = i.split()[0] + "  " + i.split()[1]
       if(temp in j):
           df.append(j)

#data3 = data2[data.columns[1:]]
remaining = []
for i in data.columns:
    if i not in df and i!=data.columns[0]:
        remaining.append(i)
t = random.sample(set(data.columns), 1248)
data3 = data[df]
data4 = data2[df]
print(len(remaining),len(df))
#data3 = data[remaining]
#data3 = data2[df]
#data4 = data2[data.columns[1:]]
train_data = data3.iloc[0:,:].values
test_data = data4.iloc[0:,:].values
y_train = data[data.columns[0]].iloc[0:].values
y_test = data2[data2.columns[0]].iloc[0:].values
train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
print(train_data.shape, y_train.shape, test_data.shape, y_test.shape)
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_data, y_train, epochs=50, batch_size=72, validation_data=(test_data, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model.predict(test_data)
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
print(mean_squared_error(y_test,yhat))


# y = data[data.columns[0]]
# x = data[df]
# x2 = data2[df]
# x3 = data3
# #model = sm.OLS(y,x)
# wls_model = sm.WLS(y,x)
# wls_results = wls_model.fit()
# pred = wls_results.predict(x2)
# y = data2[data.columns[0]]
# print("MSE with causal factors for the year 2008: ",round(mean_squared_error(y,pred),2))
#
# wls_model = sm.WLS(y,x3)
# wls_results = wls_model.fit()
# pred = wls_results.predict(data2[data.columns[1:]])
# print("MSE with all factors for the year 2008: ",round(mean_squared_error(y,pred),2))