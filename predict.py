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
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE
import numpy as np
import tensorflow as tf
from ray import tune
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import accuracy

# Mapeo de Metricas
metric_map = {
    'MAE': MAE(),
    'MAPE': MAPE(),
    'RMSE': RMSE(),
}

# Clasical

# Holt Winters
def predict_holt_winters(config=None, data=None, cutoff_date=None):
    
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=config['years'], 
                                            months_val=config['months'], 
                                            date='ds')

    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(1, 2))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    
    HoltWinters = ExponentialSmoothing(x_train['y_scaled'],
                    dates=x_train['ds'],
                    trend=config['trend_type'],  # 'add'
                    seasonal=config['seasonal_type'],  # 'add', 'mul'
                    seasonal_periods= config['seasonal_periods'],#int(info_frec[1]['periodo']),
                    damped_trend=config['damped_trend'],  # True / False
                    use_boxcox=config['use_boxcox'],
                    #freq=config['freq']
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
    unseen = data[data['ds']>cutoff_date]
    unseen['ds'] = unseen['ds'].apply(lambda x: x.replace(day=1))

    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    data['ds'] = data['ds'].apply(lambda x: x.replace(day=1))
    # Partición de Datos
    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    # Estandarización 
    # aun que inutil en el caso de XGBoost. 
    # Thus we dont use y_estandarizada
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
    x_train = utilities.exo_variables_train(train_data= x_train,
                                            lag_start=1,
                                            lag_end=48,
                                            signals=config['signals'])

    dff = utilities.exo_variables_predict(train_data= x_train, 
                                cutoff_date=x_val.ds.min(), 
                                horizon=config['h'],
                                lag_start=1,
                                lag_end=48,
                                signals=config['signals'])

    #features = list(dff.columns)[1:]
    features = ['quarter', 'month', 'year', 'dayofyear', 'dayofmonth',
       'weekofyear', 'Promedio_Historico', 'Std_Historico', 'y_last_7',
       'y_last_8', 'y_last_9', 'y_last_10', 'y_last_11', 'y_last_12',
       'y_last_13', 'y_last_14', 'y_last_15', 'y_last_16', 'y_last_17',
       'y_last_18', 'y_last_19', 'y_last_20', 'y_last_21', 'y_last_22',
       'y_last_23', 'y_last_24', 'y_last_25', 'y_last_26', 'y_last_27',
       'y_last_28', 'y_last_29', 'y_last_30', 'y_last_31', 'y_last_32',
       'y_last_33', 'y_last_34', 'y_last_35', 'y_last_36', 
       'f_seno_1', 'f_seno_2', 'f_seno_3', 'f_seno_4', 'signal', 'upper', 'lower']

    x_val = x_val.merge(dff, on=['ds'], how='inner')

    # Matrices de XGBoost
    dtrain = xgb.DMatrix(x_train[features], label=x_train['y'], feature_names=features)
    dval = xgb.DMatrix(x_val[features], label=x_val['y'], feature_names=features)

    # Matriz con toda la historia
    param = {
        'max_depth': config['max_depth'],
        'colsample_bytree': config['colsample_bytree'],
        'subsample': config['subsample'],
        'seed': 0,
        'verbosity': 0,
        'alpha': config['alpha'],
        'eta': config['eta'],
        'lambda': config['lambdaa'],
        'tree_method': 'hist',
        'eval_metric': 'mape'
    }

    # Train XGBoost model on training set
    xgb_model_train = xgb.train(
        param,
        dtrain,
        num_boost_round=config['num_boost_round'],
        early_stopping_rounds=15,
        verbose_eval=False,
        evals=[(dval, 'val')])

    dff_matrix = xgb.DMatrix(dff[features], feature_names=features)
    dff['yhat'] = xgb_model_train.predict(dff_matrix)
    # Merge con los resultados

    results = dff.merge(unseen, on=['ds'], how='inner')
    results['mape'] = results.apply(accuracy.mape, args=('yhat', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results, dff
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
                    loss=metric_map[config['metric']],
                    valid_loss=metric_map[config['metric']],
                    scaler_type='standard',
                    encoder_n_layers=config['layers'],
                    encoder_hidden_size=config['neurons'],
                    decoder_hidden_size=config['neurons'],
                    decoder_layers=config['layers'],
                    max_steps=config['max_steps'],
                    #start_padding_enabled=True
                    #futr_exog_list=['y_[lag12]'],
                    #stat_exog_list=['airline1'],
                    )
        ],
        freq=config['freq']
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
    return results, Y_hat_df
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
                    loss=metric_map[config['metric']],#DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                    valid_loss=metric_map[config['metric']],
                    learning_rate=config['learning_rate'],#0.001,
                    #stat_exog_list=['airline1'],
                    #futr_exog_list=['trend'],
                    max_steps=config['max_steps'],
                    val_check_steps=50,
                    early_stop_patience_steps=-1,
                    scaler_type='robust',
                    enable_progress_bar=False,
                    #start_padding_enabled=True
                    ),
        ],
        freq=config['freq']
    )

    nf.fit(df=x_train, target_col='y_scaled')
    Y_hat_df = nf.predict()
    Y_hat_df['lstm_v2'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    Y_hat_df['lstm_og']  = (Y_hat_df['lstm_v2']*sigma)+mu
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['lstm_og'])
    results['mape'] = results.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results, Y_hat_df
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
                    loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=False),
                    valid_loss=MQLoss(level=[80, 90]),
                    learning_rate=config['learning_rate'],#0.005,
                    #stat_exog_list=['airline1'],
                    #futr_exog_list=['trend'],
                    max_steps=config['max_steps'],
                    val_check_steps=10,
                    early_stop_patience_steps=-1,
                    scaler_type='standard',
                    start_padding_enabled=True,
                    enable_progress_bar=False,
                    ),
        ],
        freq=config['freq']
    )

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['DeepAr_v2'] = scaler.inverse_transform([Y_hat_df['DeepAR'].values])[0]
    Y_hat_df['DeepAr_og']  = (Y_hat_df['DeepAr_v2']*sigma)+mu
    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['DeepAr_og'])
    results['mape'] = results.apply(accuracy.mape, args=('DeepAr_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results, Y_hat_df
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
                                loss=metric_map[config['metric']],
                                valid_loss=metric_map[config['metric']],
                                scaler_type='robust',
                                learning_rate=1e-3,
                                max_steps=config['max_steps'],
                                val_check_steps=50,
                                early_stop_patience_steps=-1,
                                enable_progress_bar=False,
                                start_padding_enabled=True),
        ],
        freq=config['freq']
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['Transformer_v2'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
    Y_hat_df['Transformer_og']  = (Y_hat_df['Transformer_v2']*sigma)+mu
    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['Transformer_og'])
    results['mape'] = results.apply(accuracy.mape, args=('Transformer_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results, Y_hat_df
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
                    loss=metric_map[config['metric']],
                    valid_loss=metric_map[config['metric']],
                    scaler_type='robust',
                    learning_rate=config['learning_rate'],
                    max_steps=config['max_steps'],
                    val_check_steps=50,
                    start_padding_enabled=True,
                    early_stop_patience_steps=-1),
        ],
        freq=config['freq']
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
    return results, Y_hat_df
    results.to_csv('nhits_prediction.csv')