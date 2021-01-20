import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
# import standard libraries
import pandas as pd
import numpy as np
from numpy import array
from pandas_profiling import ProfileReport
from tqdm import tqdm
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys
import icecream as ic
warnings.filterwarnings('ignore')

# import time series eda libraries
from statsmodels.tsa.seasonal import seasonal_decompose

# import sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

# import keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tsmoothie.smoother import LowessSmoother
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

train_data = st.sidebar.selectbox('Which training data to use?', ["train_FD001.txt","train_FD002.txt","train_FD003.txt","train_FD004.txt"])
train_data = st.cache(pd.read_csv)('../../data/predictive-maintenance/'+train_data,sep=" ", header=None)

test_data = st.sidebar.selectbox('Which test data to use?', ["test_FD001.txt","test_FD002.txt","test_FD003.txt","test_FD004.txt"])
rul_data = st.cache(pd.read_csv)('../../data/predictive-maintenance/RUL_'+test_data[5:], sep=" ", header=None)
test_data = st.cache(pd.read_csv)('../../data/predictive-maintenance/'+test_data, sep=" ", header=None)

engines = st.sidebar.multiselect('Show different engines clubs?', train_data[0].unique())

new_df = train_data[(train_data[0].isin(engines))]
st.write(new_df)

# Create distplot with custom bin_size
fig = px.scatter(new_df,y=1,color=0)
# Plot!
st.plotly_chart(fig)

# data prep
train_data_filled=train_data.interpolate()
train_data_filled[26]=train_data_filled[26].fillna(0)
train_data_filled[27]=train_data_filled[27].fillna(0)

sc=StandardScaler()
sc.fit(train_data_filled.iloc[:,2:])
train_data_sc=train_data_filled
train_data_sc.iloc[:,2:]=sc.transform(train_data_filled.iloc[:,2:])

engines=[]
for a in train_data_sc[0].unique():
    engines.append(train_data_sc[train_data_sc[0]==a])

# for each sample, add the number of cycles left until the fault appears and separate RUL
y=[]
for e in engines:
    e.set_index(1)
    y.append(e[1].shape[0]-e[1])
    e.drop([0],axis=1,inplace=True)
    e.drop([1],axis=1,inplace=True)
    

# data smoother to treat noise

for e in engines:
    smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
    smoother.smooth(e)
    

#Select algorithm
algorithms=["Linear Regression","KNearest Neighbor","SVC","LSTM"]

algo = st.sidebar.selectbox('Which algorithm to use?', algorithms)

if algo == "Linear Regression":
    
    st.write("LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.")

    y_dataframe=pd.DataFrame(np.array([item for sublist in y for item in sublist if isinstance(item, int)]))
    X_train,X_test,y_train,y_test=train_test_split(train_data_sc.drop([0,1],axis=1),y_dataframe,train_size=0.8,shuffle=True,random_state=123)
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    ypred_tr=lr.predict(X_train)
    ypred_te=lr.predict(X_test)
    st.write("The training MSE is: ",mse(y_train,ypred_tr))
    st.write("The testing MSE is: ",mse(y_test,ypred_te))
    y_pred=lr.predict(engines[0])
    plt.plot(y[0])
    plt.plot(y_pred)

elif algo == "KNearest Neighbor":
    
    st.write("The target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.")

    y_dataframe=pd.DataFrame(np.array([item for sublist in y for item in sublist if isinstance(item, int)]))
    X_train,X_test,y_train,y_test=train_test_split(train_data_sc.drop([0,1],axis=1),y_dataframe,train_size=0.8,shuffle=True,random_state=123)
    knr = KNeighborsRegressor(n_neighbors=10)
    knr.fit(X_train, y_train)
    ypred_tr=knr.predict(X_train)
    ypred_te=knr.predict(X_test)
    st.write("The training MSE is: ",mse(y_train,ypred_tr))
    st.write("The testing MSE is: ",mse(y_test,ypred_te))
    y_pred=knr.predict(engines[0])
    plt.plot(y[0])
    plt.plot(y_pred)
elif algo == "SVR":
    
    st.write("The target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.")

    y_dataframe=pd.DataFrame(np.array([item for sublist in y for item in sublist if isinstance(item, int)]))
    X_train,X_test,y_train,y_test=train_test_split(train_data_sc.drop([0,1],axis=1),y_dataframe,train_size=0.8,shuffle=True,random_state=123)
    svr = SVR(kernel="rbf")
    svr.fit(X_train, y_train)
    ypred_tr=svr.predict(X_train)
    ypred_te=svr.predict(X_test)
    st.write("The training MSE is: ",mse(y_train,ypred_tr))
    st.write("The testing MSE is: ",mse(y_test,ypred_te))
    y_pred=svr.predict(engines[0])
    plt.plot(y[0].reset_index())
    plt.plot(y_pred)
elif algo == "LSTM":
    
    st.write("Long short-term memory is an artificial recurrent neural network architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points, but also entire sequences of data.")
    X_train,X_test,y_train,y_test=train_test_split(engines,output,train_size=0.8,random_state=123)
    for i in range(len(X_train)):
        X = []
        y = []
        datax=np.array(X_train[i])
        datay=np.array(y_train[i])
        for j in range(2, datax.shape[0]):
            X.append(datax[j-2:j, :])
            y.append(datay[j,])
    X, y = np.array(X), np.array(y).reshape(np.array(y).shape[0],1)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    y_train[i]=y
    X_train[i]=X
    for i in range(len(X_test)):
        X = []
        y = []
        datax=np.array(X_test[i])
        datay=np.array(y_test[i])
        for j in range(2, datax.shape[0]):
            X.append(datax[j-2:j, :])
            y.append(datay[j,])
    X, y = np.array(X), np.array(y).reshape(np.array(y).shape[0],1)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    y_test[i]=y
    X_test[i]=X
    
    model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = ( X_train[0].shape[1],X_train[0].shape[2])))
    model.add(Dropout(0.1))# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.1))# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.1))# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.1))# Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')
    
    # Fitting the RNN to the Training set
    for j in tqdm(range(100)): 
        for i in range(len(X_train)):
            model.train_on_batch(X_train[i], y_train[i],reset_metrics=False)

    ypred_te=model.predict(X_test[0])
    st.write("The testing MSE is: ",mse(y_test,ypred_te))
    plt.plot(y_test[0])
    plt.plot(ypred_te)

    
