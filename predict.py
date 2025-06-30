# Predict
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from utiles import utilities
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, VanillaTransformer, LSTM, RNN, NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE
import numpy as np
import tensorflow as tf
from ray import tune
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import accuracy
# Clasical

# Holt Winters
def predict_holt_winters(config=None, data=None, cutoff_date=None):
    
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['m_train'], 
                                            months_val=config['m_val'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    
    HoltWinters = ExponentialSmoothing(x_train['y_scaled'],
                    dates=x_train['ds'],
                    trend=config['trend_type'],  # 'add'
                    seasonal=config['seasonal_type'],  # 'add', 'mul'
                    seasonal_periods= int(info_frec[0]['periodo']),
                    damped_trend=config['damped_trend'],  # True / False
                    use_boxcox=config['use_boxcox'],
                    freq='W-mon'
                    ).fit()
    
    forecast = HoltWinters.forecast(steps=len(x_val))
    #Val = Val.set_index('ds')
    x_val['yhat'] = forecast
    x_val['yhat'].fillna(0, inplace=True)
    x_val['yhat'] = x_val['yhat'].clip(lower=0)
    H = pd.DataFrame()
    H['yhat'] = HoltWinters.forecast(steps=52+len(x_val))
    H = H.reset_index().rename(columns={'index':'ds'})
    # MAE
    mae = mean_absolute_error(x_val['y'], x_val['yhat'])
    # RMSE
    rmse = root_mean_squared_error(x_val['y'], x_val['yhat'])
    #print(rmse)
    return H, x_val
# Tree Methods - ML

# XGB 
def predict_xgb(config=None, data=None, cutoff_date=None):
    
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    
# Neural Network Models. 

# RNN
def predict_rnn(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    
    horizonte = config['h']
    nf = NeuralForecast(
        models=[RNN(h=horizonte,
                    input_size=horizonte*config['input_size'],
                    loss=MAE(),
                    valid_loss=MAE(),
                    scaler_type='standard',
                    encoder_n_layers=config['layers'],
                    encoder_hidden_size=config['neurons'],
                    decoder_hidden_size=config['neurons'],
                    decoder_layers=config['layers'],
                    max_steps=config['max_steps'],
                    #futr_exog_list=['y_[lag12]'],
                    #stat_exog_list=['airline1'],
                    )
        ],
        freq='W-mon'
    )
    # It trains the model
    nf.fit(df=x_train, target_col='y_scaled')
    Y_hat_df = nf.predict()
    #Y_hat_df['ds'] = Y_hat_df['ds'].apply(utilities.align_to_semi_monthly)
    Y_hat_df['rnn_v2'] = scaler.inverse_transform([Y_hat_df['RNN'].values])[0]
    Y_hat_df['rnn_og']  = (Y_hat_df['rnn_v2']*sigma)+mu

    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['rnn_og'])
    results['mape'] = results.apply(accuracy.mape, args=('rnn_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results
    results.to_csv('rnn_prediction.csv')

# LSTM
def predict_lstm(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    x_train['y_scaled'] = data_scaled
    horizonte = config['h']
    nf = NeuralForecast(
        models=[LSTM(h=horizonte,
                    #input_size=24,
                    input_size=horizonte*config['input_size'],
                    encoder_n_layers=config['layers'],
                    encoder_hidden_size=config['neurons'],
                    decoder_hidden_size=config['neurons'],
                    decoder_layers=config['layers'],
                    loss=MAE(),#DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                    valid_loss=RMSE(),
                    learning_rate=config['learning_rate'],#0.001,
                    #stat_exog_list=['airline1'],
                    #futr_exog_list=['trend'],
                    max_steps=config['max_steps'],
                    val_check_steps=50,
                    early_stop_patience_steps=-1,
                    scaler_type='robust',
                    enable_progress_bar=False,
                    ),
        ],
        freq='W-mon'
    )

    nf.fit(df=x_train, target_col='y_scaled')
    Y_hat_df = nf.predict()
    Y_hat_df['lstm_v2'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    Y_hat_df['lstm_og']  = (Y_hat_df['lstm_v2']*sigma)+mu
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['lstm_og'])
    results['mape'] = results.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results
    results.to_csv('lstm_prediction.csv')

# DeepAr
def predict_deepAr(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]    
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    horizonte = config['h']
    nf = NeuralForecast(
        models=[DeepAR(h=horizonte,
                    input_size=horizonte*config['input_size'],
                    lstm_n_layers=config['layers'],
                    trajectory_samples=config['trajectories'],
                    lstm_hidden_size=config['neurons'],
                    loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                    valid_loss=MQLoss(level=[80, 90]),
                    learning_rate=config['learning_rate'],#0.005,
                    #stat_exog_list=['airline1'],
                    #futr_exog_list=['trend'],
                    max_steps=config['max_steps'],
                    val_check_steps=10,
                    early_stop_patience_steps=-1,
                    scaler_type='standard',
                    enable_progress_bar=False,
                    ),
        ],
        freq='W-mon'
    )

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['DeepAr_v2'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    Y_hat_df['DeepAr_og']  = (Y_hat_df['DeepAr_v2']*sigma)+mu
    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['DeepAr_og'])
    results['mape'] = results.apply(accuracy.mape, args=('DeepAr_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results
    results.to_csv('deep_ar_prediction.csv')

# Transformer
def predict_transformer(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    horizonte = config['h']
    nf = NeuralForecast(
        models=[VanillaTransformer(h=horizonte,
                                input_size=horizonte*config['input_size'],
                                hidden_size=config['neurons'],
                                conv_hidden_size=config['conv_size'],
                                n_head=config['n_heads'],
                                loss=MAE(),
                                valid_loss=RMSE(),
                                scaler_type='robust',
                                learning_rate=1e-3,
                                max_steps=config['max_steps'],
                                val_check_steps=50,
                                early_stop_patience_steps=-1,
                                enable_progress_bar=False),
        ],
        freq='W-mon'
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['Transformer_v2'] = scaler.inverse_transform([Y_hat_df['Transformer'].values])[0]
    Y_hat_df['Transformer_og']  = (Y_hat_df['Transformer_v2']*sigma)+mu
    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['Transformer_og'])
    results['mape'] = results.apply(accuracy.mape, args=('Transformer_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results
    results.to_csv('transformer_prediction.csv')

# NHITS
def predict_nhits(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled

    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    horizonte = config['h']
    nf = NeuralForecast(
        models=[NHITS(h=horizonte,
                    input_size=horizonte*config['input_size'],
                    #hidden_size=config['neurons'],
                    n_freq_downsample=config['n_freq_downsample'],
                    n_pool_kernel_size = config['n_pool_kernel_size'],
                    n_blocks=[1,1,1],
                    mlp_units=3*[[config['neurons'], config['neurons']]],
                    loss=MAE(),
                    valid_loss=RMSE(),
                    scaler_type='robust',
                    learning_rate=config['learning_rate'],
                    max_steps=config['max_steps'],
                    val_check_steps=50,
                    early_stop_patience_steps=-1),
        ],
        freq='W-mon'
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['nhits_v2'] = scaler.inverse_transform([Y_hat_df['NHITS'].values])[0]
    Y_hat_df['nhits_og']  = (Y_hat_df['nhits_v2']*sigma)+mu
    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['nhits_og'])
    results['mape'] = results.apply(accuracy.mape, args=('nhits_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results
    results.to_csv('nhits_prediction.csv')