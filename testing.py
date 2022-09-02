import pandas as pd
import numpy as np
import random

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
from statsmodels.tsa.vector_ar.var_model import VAR
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.svm import SVR


from numpy.random import seed
seed(100)
import tensorflow
tensorflow.random.set_seed(100)

random.seed(100)
np.random.seed(100)
PYTHONHASHSEED=100
def forecast_accuracy(forecast, actual):
    mape = 1/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))  # MAPE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.abs(np.mean((forecast - actual)/actual))   # MPE
    mse = np.mean((forecast - actual)**2)  # MSE
    rae = np.sum((forecast - actual)**2)**0.5/np.sum(actual**2)**0.5
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'mae': mae,
            'mpe': mpe, 'mse':mse, 'minmax':minmax,'rae':rae})


data = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/Yearly_Data_Normalized/FilteredNormalized2007.csv")
data2 = pd.read_csv("/home/local/ASUAD/psheth5/RA-Hydrological/Yearly_Data_Normalized/FilteredNormalized2006.csv")
t_causes = pd.read_csv("CausesofTargetNode2.csv")
st_causes = pd.read_csv("CausesofTargetNode.csv")
x = t_causes.groupby("to")
df = []
for i in x.get_group("-95.35  28.85")['from'].values:
   for j in data.columns:
       temp = i.split()[0] + "  " + i.split()[1]
       if(temp in j):
           df.append(j)

x = st_causes.groupby("to")
df1 = []
for i in x.get_group("-95.35  28.85")['from'].values:
   for j in data.columns:
       temp = i.split()[0] + "  " + i.split()[1]
       if(temp in j):
           df1.append(j)

remaining = []
for i in data.columns:
    if i not in df and i!=data.columns[0]:
        remaining.append(i)
t = random.sample(set(data.columns), 1248)

#Choosing the features to use

#all features
# data3 = data[data.columns[1:]]
# data4 = data2[data.columns[1:]]

#random_features
#data3 = data[t]
#data4 = data2[t]

# #remaining_features
# data3 = data[remaining]
# data4 = data2[remaining]

# #st_features
# data3 = data[df1]
# data4 = data2[df1]

#t_features
data3 = data[df]
data4 = data2[df]
c=0
for i in range(len(t)):
    if(t[i] in df):
        c+=1
print(round(c/len(t),2))

train_data = data3.iloc[0:334,:].values
test_data = data3.iloc[334:,:].values
y_train = data[data.columns[0]].iloc[0:334].values
y_test = data[data.columns[0]].iloc[334:].values
train_data1 = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
test_data1 = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

#LSTM MODEL
model = Sequential()
model.add(LSTM(50, input_shape=(train_data1.shape[1], train_data1.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(train_data1, y_train, epochs=100, batch_size=72, verbose=2)
# make a prediction
yhat = model.predict(test_data1)
y_test1 = y_test.reshape(y_test.shape[0],1)
accuracy_prod = forecast_accuracy(yhat, y_test1)
print("FOR LSTM MODEL")
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


#CNN MODEL
model = Sequential()
model.add(Conv1D(filters=50, kernel_size=2, activation='relu', padding='same', input_shape=(train_data1.shape[1], train_data1.shape[2])))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_data1, y_train, epochs=10, batch_size=91, verbose=2)
# make a prediction
yhat = model.predict(test_data1)
y_test1 = y_test.reshape(y_test.shape[0],1)
print("FOR CNN MODEL")
accuracy_prod = forecast_accuracy(yhat, y_test1)
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


#MLP MODEL
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=train_data.shape[1]))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_data, y_train, epochs=100, batch_size=72, verbose=2)
# make a prediction
yhat = model.predict(test_data)
print("FOR MLP MODEL")
y_test1 = y_test.reshape(y_test.shape[0],1)
accuracy_prod = forecast_accuracy(yhat, y_test1)
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))

#SVR
svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
svr_rbf.fit(train_data, y_train)
yhat = svr_rbf.predict(test_data)
accuracy_prod = forecast_accuracy(yhat, y_test)
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


#RFR
mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
mdl.fit(train_data,y_train)
yhat = mdl.predict(test_data)
accuracy_prod = forecast_accuracy(yhat, y_test)
for k, v in accuracy_prod.items():
    print(k, ': ', round(v, 4))

