# !pip install plotly shap

# packages
import time
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.dates as mdates
import collections
import ast
import seaborn as sns
import scipy.stats as stats
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.ndimage.filters import gaussian_filter
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime,timedelta
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost
import os
import shap
import tensorflow as tf

from utils import calc_metrics, extract_ts_features, generateValSet, plot_forecast, return_metrics
tf.compat.v1.disable_v2_behavior()

import warnings
warnings.filterwarnings('ignore')

# Data Loading

df_mt3 = pd.read_csv('/data/electricity_321.csv', parse_dates={'timestamp' : [0]}, infer_datetime_format=True) 
df_mt3 = df_mt3.iloc[:, [0,4]]

df_mt3 = df_mt3.dropna()
df_mt3 = df_mt3[df_mt3.MT_3 != 0]

df_mt3 = extract_ts_features(df_mt3)

data_split = 0.85

Dataset = df_mt3.copy()
Dataset_time = df_mt3['timestamp']

df_mt3.drop('timestamp', axis=1, inplace=True)

## LSTM

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(df_mt3.iloc[:,1:])
Y_data = Y_scaler.fit_transform(df_mt3['MT_3'].to_numpy().reshape(-1,1))

data_series = Dataset["MT_3"].copy()
n_train = int(len(df_mt3) * data_split)
n_test = (len(df_mt3) - n_train)
print(n_train, n_test)

X_train = X_data[:-n_test]
y_train = Y_data[:-n_test]
train_dates_lstm = Dataset_time[:-n_test]

X_test = X_data[-n_test:]
y_test_LSTM = Y_data[-n_test:]
test_dates_lstm = Dataset_time[-n_test:]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape, y_train.shape, X_test.shape, y_test_LSTM.shape)

model = Sequential()
model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
model.summary()

batch_size = 256
buffer_size = 150
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test_LSTM))
val_data = val_data.batch(batch_size).repeat()

model_path = 'LSTM_Multivariate_No_Lag.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint] 
history = model.fit(train_data,epochs=100,steps_per_epoch=100,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)

plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show()

model = load_model('LSTM_Multivariate_No_Lag.h5')
model.evaluate(X_test, y_test_LSTM)

y_pred_LSTM = model.predict(X_test)
pred_Inverse_LSTM = Y_scaler.inverse_transform(np.reshape(y_pred_LSTM, (y_pred_LSTM.shape[0], y_pred_LSTM.shape[1])))
y_test_inverse_LSTM = np.array(df_mt3.iloc[-n_test:,0])

### Deep SHAP

explainer = shap.DeepExplainer(model, X_train[:1000])
shap_values = explainer.shap_values(X_test[:100], check_additivity=False)

shap.summary_plot(np.array(shap_values[0]).reshape(100,9), features=np.array(X_test[:100]).reshape(100,9), feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'])

shap.summary_plot(np.array(shap_values[0]).reshape(100,9), features=np.array(X_test[:100]).reshape(100,9), feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'], plot_type='bar')

## GRU

model = Sequential()
model.add(GRU(200, activation='relu', return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(GRU(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
model.summary()

batch_size = 256
buffer_size = 150
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test_LSTM))
val_data = val_data.batch(batch_size).repeat()

model_path = 'GRU_Multivariate_No_Lag.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint] 
history = model.fit(train_data,epochs=100,steps_per_epoch=100,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)

plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show()

model = load_model('GRU_Multivariate_No_Lag.h5')
model.evaluate(X_test, y_test_LSTM)

y_pred_GRU = model.predict(X_test)
pred_Inverse_GRU = Y_scaler.inverse_transform(np.reshape(y_pred_GRU, (y_pred_GRU.shape[0], y_pred_GRU.shape[1])))
y_test_inverse_GRU = np.array(df_mt3.iloc[-n_test:,0])

### Deep SHAP

explainer = shap.DeepExplainer(model, X_train[:1000])
shap_values = explainer.shap_values(X_test[:100], check_additivity=False)

shap.summary_plot(np.array(shap_values[0]).reshape(100,9), features=np.array(X_test[:100]).reshape(100,9), feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'])

shap.summary_plot(np.array(shap_values[0]).reshape(100,9), features=np.array(X_test[:100]).reshape(100,9), feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'], plot_type='bar')

## RF

data_series = Dataset["MT_3"].copy()
n_train = int(len(Dataset.MT_3) * data_split)
n_test = (len(Dataset) - n_train)
# look_back = 2
n_train, n_test

# creating target and features for training set
X_train = np.array(df_mt3[:-n_test].loc[:, df_mt3.columns != 'MT_3'])
y_train = np.array(df_mt3[:-n_test].iloc[:, 0])
train_dates = Dataset_time[:-n_test]

# creating target and features for test set
X_test = np.array(df_mt3[-n_test:].loc[:, df_mt3.columns != 'MT_3'])
y_test_RF = np.array(df_mt3[-n_test:].iloc[:, 0])
test_dates = Dataset_time[-n_test:]

print(X_train.shape, y_train.shape, X_test.shape, y_test_RF.shape)

RF_Model1 = RandomForestRegressor(n_estimators=1000, max_features=1, random_state=123)

labels = y_train
features = X_train
 
# Fit the RF model with features and labels.
rgr=RF_Model1.fit(X_train, y_train)
 
# Now that we've run our models and fit it, let's create
# dataframes to look at the results
predictions_RF=rgr.predict(X_test)

# plot of predictions and actual values
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=y_test_RF, line_shape='linear', 
              name = 'Ground Truth'))
fig.add_trace(go.Scatter(x=test_dates, y=predictions_RF, line_shape='linear', 
              name = 'Prediction'))
fig.show()    

# calculating RMSE metrics
error = np.sqrt(mean_squared_error(y_test_RF[:-1], y_test_RF[1:]))
print('Baseline RMSE: %.3f' % error)
error = np.sqrt(mean_squared_error(predictions_RF, y_test_RF))
print('Test RMSE: %.3f' % error)
print(error / np.mean(y_test_RF))

features = df_mt3.columns[df_mt3.columns != 'MT_3']
importances = rgr.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

## GBR

GBR_Model1 = GradientBoostingRegressor(n_estimators=3000, max_features='auto', criterion='squared_error', random_state=123)

# Fit the RF model with features and labels.
gbr=GBR_Model1.fit(X_train, y_train)
 
# Now that we've run our models and fit it, let's create
# dataframes to look at the results
predictions_GBR=gbr.predict(X_test)

# plot of predictions and actual values
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=y_test_RF, line_shape='linear', 
              name = 'Ground Truth'))
fig.add_trace(go.Scatter(x=test_dates, y=predictions_GBR, line_shape='linear', 
              name = 'Prediction'))
fig.show()    

# calculating RMSE metrics
error = np.sqrt(mean_squared_error(y_test_RF[:-1], y_test_RF[1:]))
print('Baseline RMSE: %.3f' % error)
error = np.sqrt(mean_squared_error(predictions_GBR, y_test_RF))
print('Test RMSE: %.3f' % error)
print(error / np.mean(y_test_RF))

features = df_mt3.columns[df_mt3.columns != 'MT_3']
importances = gbr.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

## XGB

XGB_Model1 = xgboost.XGBRegressor(n_estimators=1000, max_features=1, random_state=123)

labels = y_train
features = X_train
 
# Fit the RF model with features and labels.
xgb=XGB_Model1.fit(X_train, y_train)
 
# Now that we've run our models and fit it, let's create
# dataframes to look at the results
predictions_XGB=xgb.predict(X_test)

# plot of predictions and actual values
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=y_test_RF, line_shape='linear', 
              name = 'Ground Truth'))
fig.add_trace(go.Scatter(x=test_dates, y=predictions_XGB, line_shape='linear', 
              name = 'Prediction'))
fig.show()    

# calculating RMSE metrics
error = np.sqrt(mean_squared_error(y_test_RF[:-1], y_test_RF[1:]))
print('Baseline RMSE: %.3f' % error)
error = np.sqrt(mean_squared_error(predictions_XGB, y_test_RF))
print('Test RMSE: %.3f' % error)
print(error / np.mean(y_test_RF))

features = df_mt3.columns[df_mt3.columns != 'MT_3']
importances = xgb.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

## Evaluation

len(test_dates), len(y_test_RF), len(predictions_RF), len(predictions_GBR), len(predictions_XGB), len(pred_Inverse_LSTM), len(pred_Inverse_GRU)

test_dates = test_dates.reset_index(drop='True')
test_dates.index

pd.concat([pd.DataFrame(list(test_dates), columns=['Timestamp']),
           pd.DataFrame(y_test_RF, columns=['Ground Truth']), 
           pd.DataFrame(predictions_GBR, columns=['GBR']), 
           pd.DataFrame(predictions_RF, columns=['RF']), 
           pd.DataFrame(predictions_XGB, columns=['XGB']),
           pd.DataFrame(pred_Inverse_LSTM, columns=['LSTM']),
           pd.DataFrame(pred_Inverse_GRU, columns=['GRU'])], axis=1).to_csv('sample_results/no_lag_results.csv', index=False)
df_res = pd.DataFrame(columns=['model', 'hours', 'mape', 'mae', 'rmse', 'nrmse'])
df_res = df_res.astype(dtype= {"model":"string",
        "hours":"int","mape":"string", "mae":"string", "rmse":"string", "nrmse":"string"})
pred_hours = [1,6,12,24,48]

for hours in pred_hours:
  for model_name in ['RF', 'GBR', 'XGB', 'LSTM', 'GRU']:
    Mape = []
    Mae = []
    Rmse = []
    Nrmse = []
    
    val_set = None

    if model_name == 'RF':
      val_set = generateValSet(len(y_test_RF), hours)
      holdout = predictions_RF
      ground_truth = y_test_RF
    elif model_name == 'GBR':
      val_set = generateValSet(len(y_test_RF), hours)
      holdout = predictions_GBR
      ground_truth = y_test_RF
    elif model_name == 'XGB':
      val_set = generateValSet(len(y_test_RF), hours)
      holdout = predictions_XGB
      ground_truth = y_test_RF
    elif model_name=='LSTM':
      val_set = generateValSet(len(y_test_LSTM), hours)
      holdout = pred_Inverse_LSTM
      ground_truth = y_test_inverse_LSTM
    else:
      val_set = generateValSet(len(y_test_LSTM), hours)
      holdout = pred_Inverse_GRU
      ground_truth = y_test_inverse_GRU

    for val in val_set:
      forecast = holdout[:val]
      pred = forecast[-hours:]
      true = ground_truth[:val]
      actual = true[-hours:]
      mae, mape, rmse, nrmse = calc_metrics(actual, pred)
      Mape.append(mape)
      Mae.append(mae)
      Rmse.append(rmse)
      Nrmse.append(nrmse)
      
    metrics = return_metrics(Mape, Mae, Rmse, Nrmse)
    df_res = df_res.append({'model':model_name, 'hours': hours, 'mape': metrics[0], 'mae': metrics[1], 'rmse': metrics[2], 'nrmse': metrics[3]}, ignore_index=True)

df_res.to_csv('sample_results/no_lag_results_metrics.csv', index=False)

### 6 Hour Window

plot_forecast(11,6, y_test_RF, predictions_RF, predictions_GBR, predictions_XGB, pred_Inverse_LSTM, pred_Inverse_GRU)

### 12 Hour Window

plot_forecast(24,12, y_test_RF, predictions_RF, predictions_GBR, predictions_XGB, pred_Inverse_LSTM, pred_Inverse_GRU)

### 24 Hour Window

plot_forecast(11,24, y_test_RF, predictions_RF, predictions_GBR, predictions_XGB, pred_Inverse_LSTM, pred_Inverse_GRU)

### 48 Hour Window

plot_forecast(10,48, y_test_RF, predictions_RF, predictions_GBR, predictions_XGB, pred_Inverse_LSTM, pred_Inverse_GRU)

## Tree SHAP - GBR

explainer = shap.TreeExplainer(model=gbr,
                               data=None,
                               model_output='raw',
                               feature_perturbation='tree_path_dependent')

shap_values = explainer.shap_values(X_test)

print(f'Shape of test dataset: {X_test.shape}')
print(f'Type of shap_values: {type(shap_values)}. Length of the list: {len(shap_values)}')
print(f'Shape of shap_values: {np.array(shap_values).shape}')

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], features=X_test[0,:], feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'])

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[:100], features=X_test[:100,:], feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'])

shap.summary_plot(shap_values, features=X_test, feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'])

shap.summary_plot(shap_values, features=X_test, feature_names=df_mt3.columns[df_mt3.columns != 'MT_3'], plot_type='bar')