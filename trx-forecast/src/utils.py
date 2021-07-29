import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from statsmodels.tsa.arima.model import ARIMA
from scipy.special import inv_boxcox

def split_data(data, lag_test):
    """
    This function splits the data in train and test sets
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contains the transactional data.
    lag_test : int
        Amount of days to test set
    
    Returns
    -------
    train, test : pandas.DataFrame
        train and test sets based on the last t days
    """
    X = data.copy()

    unique_days = X['fecha'].str[:10].unique()
    unique_days = sorted(unique_days)
    start_test  = unique_days[-lag_test]
    end_test    = unique_days[-1]
    
    print('Test dataset is from {} to {}'.format(start_test, end_test))
    
    train = X[X['fecha'] < start_test]
    test  = X[X['fecha'] >= start_test]
    
    return train, test

def process_data(data):
    """
    This function aims to procees the transactional data.
    First, It computes the transactions' number by day, oper and idTerminal.
    Seconf, It builds features like day name and if the day is weekend or not.
    
    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame contains the transactional data.
    
    Returns
    -------
    data_diaria : pandas.DataFrame
        DataFrame contains the processed data.
    """
    X = data.copy()
    
    X['dia'] = X['fecha'].str[:10]
    
    data_diaria = X.groupby(['dia', 'oper', 'idTerminal']).size()
    data_diaria = pd.DataFrame(data_diaria)
    data_diaria.columns = ['num_trx']
    data_diaria = data_diaria.reset_index()
    
    # Holidays
    co_holidays = holidays.CO()
    holidays_serie = data_diaria['dia'].apply(lambda x: co_holidays.get(x))
    
    # Additional features
    weekDays = ("Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom")

    data_diaria['dia'] = pd.to_datetime(data_diaria['dia'])
    data_diaria['nombre_dia'] = data_diaria['dia'].apply(lambda x: weekDays[x.weekday()])
    data_diaria['weekend'] = np.where(data_diaria['nombre_dia'].isin(['Sab', 'Dom']), 1, 0)
    data_diaria['weekend'] = np.where(holidays_serie.isnull(), data_diaria['weekend'], 1)
    
    return data_diaria

# NAIVE Model
def train_naive(x):
    """
    This function trains the naive model
    
    Parameters
    ----------
    x : numpy.array, pandas.Series
        Training series
    
    Returns
    -------
    model : pandas.Series
        7 lags computed
    """
    x = pd.Series(x)
    model = x.shift(7)
    
    return model

def predict_naive(model, periods):
    """
    This function forecast for the next periods.
    
    Parameters
    ----------
    model : pandas.Series
        7 lags for each date.
    periods : int
        Number of periods to forecast.
        
    Returns
    -------
    predictions : list
        Predictions for the next periods.
    """
    model = model.tolist()
    predictions = []
    pred_remain = periods
    
    if pred_remain >= 7:
        pred_batch = model[-7:]
    else:
        pred_batch = model[-pred_remain:]
    
    pred_remain = pred_remain - 7
    predictions += pred_batch
    
    while pred_remain > 0:
        
        if pred_remain >= 7:
            pred_batch = predictions[-7:]
        else:
            pred_batch = predictions[:pred_remain]
        
        predictions += pred_batch
        pred_remain = pred_remain - 7
    
    return predictions

# ARIMA Model
def train_arima(x, p, d, q):
    """
    This function trains an ARIMA model
    
    Parameters
    ----------
    x : numpy.array, pandas.Series
        Training series
    p : int
        Autoregressive order
    d : int
        Diferenciation order
    q : int
        Moving mean order
    
    Returns
    -------
    model_fit : stats.model
        Model trained
    """
    model = ARIMA(x, order = (p, d, q))
    model_fit = model.fit()
    
    return model_fit

def predict_arima(model, periods, lmbda = None):
    """
    This function forecast for the next periods.
    
    Parameters
    ----------
    model : pandas.Series
        ARIMA model
    periods : int
        Number of periods to forecast.
    lmbda : float (default = None)
        Lambda obtained from Box-Cox approach
        
    Returns
    -------
    predictions : list
        Predictions for the next periods.
    """
    predictions = model.forecast(steps = periods)
    
    if lmbda is not None:
        predictions = inv_boxcox(predictions, lmbda)
    
    return predictions

def predict_regression(x, model):
    """
    This function forecast for the next periods.
    
    Parameters
    ----------
    x : pandas.DataFrame
        DataFrame with the test information to predict
    model : sklearn.model
        sklearn model trained
        
    Returns
    -------
    predictions : list
        Predictions for the next periods.
    """
    predictions = []
    for i in range(len(x)):

        row_x = x.iloc[i, :]

        if np.isnan(row_x['lag1']):
            row_x['lag1'] = predictions[-1]

        if np.isnan(row_x['lag7']):
            row_x['lag7'] = predictions[-7]

        row_x = row_x.values.reshape(1, -1)

        pred_x = model.predict(row_x)
        pred_x = float(pred_x)

        predictions.append(pred_x)
    
    return predictions
