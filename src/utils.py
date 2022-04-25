import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go

# Extracts features and returns new dataframe
def extract_ts_features(df, primary=True, cyclic=True):
  if primary:
    df['hour'] = [i.hour for i in df['timestamp']]
    df['month'] = [i.month for i in df['timestamp']]
    df['year'] = [i.year for i in df['timestamp']]
    df['day_of_week'] = [i.dayofweek for i in df['timestamp']]
    df['day_of_year'] = [i.dayofyear for i in df['timestamp']]

  if cyclic:
    # turn time data to be cyclic
    df['dow_sin'] = np.sin(df.day_of_week * (2 * np.pi / 7))
    df['dow_cos'] = np.cos(df.day_of_week * (2 * np.pi / 7))
    df['hour_sin'] = np.sin(df.hour * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df.hour * (2 * np.pi / 24))

  return df

# Calculates evaluation metrics for given true and predicted values
def calc_metrics(y_true, y_pred): 
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    nrmse = rmse / (np.mean(y_true))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mape, rmse, nrmse

# Returns metrics averaged for a given prediction horizon
def return_metrics(Mape, Mae, Rmse, Nrmse):
  mape = str(round(np.mean(Mape), 2))+'%'+' +- '+str(round(np.std(Mape),2)) 
  mae = str(round(np.mean(Mae), 2)) + ' +- ' + str(round(np.std(Mae),2))
  rmse = str(round(np.mean(Rmse), 2)) + ' +- ' + str(round(np.std(Rmse),2))
  nrmse = str(round(np.mean(Nrmse), 2)) + ' +- ' + str(round(np.std(Nrmse),2))
  return mape, mae, rmse, nrmse

# Generates the horizon samples for a given test size
# Example I/P: Test size = 15% of dataset let's say 120. Sample_size = 6 hour window.
# O/P: [6,12,18,24....114,120] i.e. 20 samples of 6 
def generateValSet(test_size, sample_size):
  val_set = []
  i = 0
  while i < test_size:
    i += sample_size
    if i > test_size:
      i = test_size
      val_set.append(i)
      break
    val_set.append(i)
  return val_set

# Plots ground truth vs predictions of all the models
# Interval: Any sample between 1-20 if we consider the above example of generateValSet
# Hours: Prediction Horizon to be visualized
def plot_forecast(interval:int, hours:int, y_test, predictions_RF, predictions_GBR, predictions_XGB, pred_Inverse_LSTM, pred_Inverse_GRU):

  val_set = generateValSet(len(y_test), hours)
  if interval < 0 or interval > len(val_set):
    print(f"Please select between 1 and {len(val_set)}")

  low, high = val_set[interval-1]-hours, val_set[interval-1]

  forecast_rf = predictions_RF[low:high]
  forecast_gbr = predictions_GBR[low:high]
  forecast_xgb = predictions_XGB[low:high]
  forecast_lstm = [i[0] for i in pred_Inverse_LSTM[low:high]]
  forecast_gru = [i[0] for i in pred_Inverse_GRU[low:high]]
  true = y_test[low:high]

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=[i for i in range(1, hours+1)], y=true, mode='lines', line=dict(color='green'),
                name = 'Ground Truth'))
  fig.add_trace(go.Scatter(x=[i for i in range(1, hours+1)], y=forecast_rf, mode='lines', line=dict(color='blue'),
                name = 'RF'))
  fig.add_trace(go.Scatter(x=[i for i in range(1, hours+1)], y=forecast_gbr, mode='lines', line=dict(color='red'),
                name = 'GBR'))
  fig.add_trace(go.Scatter(x=[i for i in range(1, hours+1)], y=forecast_xgb, mode='lines', line=dict(color='orange'),
                name = 'XGB'))
  fig.add_trace(go.Scatter(x=[i for i in range(1, hours+1)], y=forecast_lstm, mode='lines', line=dict(color='yellow'),
                name = 'LSTM'))
  fig.add_trace(go.Scatter(x=[i for i in range(1, hours+1)], y=forecast_gru, mode='lines', line=dict(color='gray'),
                name = 'GRU'))
  
  fig.show()
  # pio.write_image(fig, "Images/PredictionVsGroundTruth.pdf", width=1000)