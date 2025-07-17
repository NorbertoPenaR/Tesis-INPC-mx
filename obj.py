# Obj Functions
# tune Example 
from ray import tune
from functools import partial
from sklearn.preprocessing import MinMaxScaler

'''
def objective(config, data):
    x, y = data
    # aquí puedes usar los datos para entrenar un modelo con los hiperparámetros de config
    score = config["a"] ** 2 + config["b"] + sum(x) + sum(y)  # solo un ejemplo
    return {"score": score}

data = ([1, 2, 3], [4, 5, 6])  # ejemplo de datos

# usamos partial para fijar los datos como argumento adicional
tuner = tune.Tuner(
    partial(objective, data=data),
    param_space={
        "a": tune.grid_search([0.001, 0.01]),
        "b": tune.choice([1, 2]),
    }
)

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)
'''
## 
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
from neuralforecast.auto import AutoNHITS
# 
import accuracy

# Clasical Models
import os
import pytorch_lightning as pl
trainer = pl.Trainer(logger=False, enable_progress_bar=False)

import logging
import torch
import warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = CRITICAL
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
from pytorch_lightning.utilities import rank_zero
rank_zero._should_log = False  # disables rank_zero_info and similar


# Lo ideal es que se pueda pasar a la construcción la metrica 
# a minimizar, la frequencia con la q se quiere predecir
# Dado el agrupamiento de los datos.

# Mapeo de Metricas
metric_map = {
    'MAE': MAE(),
    'MAPE': MAPE(),
    'RMSE': RMSE(),
}

def obj_holt_winters(config=None, data=None):

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
    #Train = x_train.copy()
    #Train['ds'] = pd.to_datetime(Train['ds'])
    features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])

    '''print(x_train)
    print('Estaconalidad')
    print(int(info_frec[0]['periodo']))
    print(info_frec)'''
    HoltWinters = ExponentialSmoothing(x_train['y_scaled'],
                    dates=x_train['ds'],
                    trend=config['trend_type'],  # 'add'
                    seasonal=config['seasonal_type'],  # 'add', 'mul'
                    seasonal_periods= config['seasonal_periods'],#int(info_frec[1]['periodo']),
                    damped_trend=config['damped_trend'],  # True / False
                    use_boxcox=config['use_boxcox'],
                    #freq=config['freak']
                    ).fit()
    
    forecast = HoltWinters.forecast(steps=len(x_val))
    #Val = Val.set_index('ds')
    x_val['yhat'] = forecast
    x_val['yhat'].fillna(0, inplace=True)
    x_val['yhat'] = x_val['yhat'].clip(lower=0)
    #Val['yhat'] = Val['yhat'].astype(int)
    
    # MAE
    mae = mean_absolute_error(x_val['y'], x_val['yhat'])
    # RMSE
    rmse = root_mean_squared_error(x_val['y'], x_val['yhat'])
    # Send the current training result back to Tune
    tune.report({"rmse": rmse})

## Seasonal Naive

# Machine Learning

# XGBoost
# Hacer Distintos Experimentos para evaluar
# Si combiene meter más variables exogenas
# Que en este caso serán variables senoidales
# O si es mejor dejarlo así. Osea, un A/B Test.
def obj_xgb(config=None, data=None):

    data['ds'] = data['ds'].apply(lambda x: x.replace(day=1))

    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    #data = utilities.features_from_date(data, ds='ds')
    
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


    features = list(dff.columns)[1:]
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

    # Evaluate model on test set and return score
    x_val['yhat'] = xgb_model_train.predict(dval)
    x_val['mape'] = x_val.apply(accuracy.mape, args=('yhat','y'), axis=1)
    mape_total = x_val['mape'].mean()
    # MAE
    mae = mean_absolute_error(x_val['y'], x_val['yhat'])
    # RMSE
    rmse = root_mean_squared_error(x_val['y'], x_val['yhat'])

    if config['metric'] == 'MAPE':
        tune.report({"rmse": mape_total})
    else:
        tune.report({"rmse": mae})

# Neural Networks

# Se usa Nixtla por que está tan bien implementado y optimizado, que realmente se pueden obtener 
# Buenos resultados a partir de la construcción de modelos.
# Además de ahorrar tiempo, y posibles sesgos al construir las ventanas de tiempo.
# al final del día, fue construido para acelerar procesos y no usarlo sería ilogico. 
# RNN
def obj_rnn(config=None, data=None):
    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    # Estandarización.
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    # Escalado de Datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled

    horizonte = config['h']
    nf = NeuralForecast(
        models=[RNN(h=horizonte,
                    input_size=horizonte*config['input_size'],
                    # Metricas de Evaluación
                    loss=metric_map[config['metric']],
                    valid_loss=metric_map[config['metric']],
                    # Escalamiento de Datos & Posibles variables exogenas
                    scaler_type='standard',
                    encoder_n_layers=config['layers'],
                    encoder_hidden_size=config['neurons'],
                    decoder_hidden_size=config['neurons'],
                    decoder_layers=config['layers'],
                    max_steps=config['max_steps'],
                    #futr_exog_list=['senoidales'],
                    #stat_exog_list=['airline1'],
                    enable_progress_bar=False,
                    #start_padding_enabled=True
                    )
        ],
        freq=config['freq']
    )

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=0)
    #Y_hat_df['ds'] = Y_hat_df['ds'].apply(utilities.align_to_semi_monthly)
    Y_hat_df['rnn_v2'] = scaler.inverse_transform([Y_hat_df['RNN'].values])[0]
    Y_hat_df['rnn_og']  = (Y_hat_df['rnn_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['rnn_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['rnn_og'])
    # MAPE
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('rnn_og', 'y'), axis=1)

    total_mape = comparativa['mape'].mean()
    if config['metric'] == 'MAPE':
        tune.report({"rmse": total_mape})
    else:
        tune.report({"rmse": mae})

# LSTM
def obj_lstm(config=None, data=None):

    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Para desescalar después:
    #original = scaler.inverse_transform(data_scaled)
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
                    # Metricas
                    loss=metric_map[config['metric']],#DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                    valid_loss=metric_map[config['metric']],
                    # Learning Rate
                    learning_rate=config['learning_rate'],#0.001,
                    #stat_exog_list=['tipo'],
                    #futr_exog_list=['senoidales'],
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

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['lstm_v2'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    Y_hat_df['lstm_og']  = (Y_hat_df['lstm_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)
    # MAPE
    total_mape = comparativa['mape'].mean()
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['lstm_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['lstm_og'])

    if config['metric'] == 'MAPE':
        tune.report({"rmse": total_mape})
    else:
        tune.report({"rmse": mae})
    # WE are returning MAE. 
    # Objetive Funtion Construcction must Match 
    # The construcción of the Predict Function.
    # a slight change, such as the loss funcion ie. MAE, RMSE
    # Will cause totally different results.

# DeepAr
def obj_deep_ar(config=None, data=None):
    
    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    # Estandarización
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    # Escalado de datos a [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    x_train['y_scaled'] = data_scaled
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
                    enable_progress_bar=False,
                    start_padding_enabled=True
                    ),
        ],
        freq=config['freq']
    )

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['DeepAr_v2'] = scaler.inverse_transform([Y_hat_df['DeepAR'].values])[0]
    Y_hat_df['DeepAr_og']  = (Y_hat_df['DeepAr_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('DeepAr_og', 'y'), axis=1)
    # MAPE
    total_mape = comparativa['mape'].mean()
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['DeepAr_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['DeepAr_og'])
    
    if config['metric'] == 'MAPE':
        tune.report({"rmse": total_mape})
    else:
        tune.report({"rmse": rmse})
    #MAPE

# Transformer
def obj_transformer(config=None, data=None):
    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Para desescalar después:
    #original = scaler.inverse_transform(data_scaled)
    x_train['y_scaled'] = data_scaled
    horizonte = config['h']
    nf = NeuralForecast(
        models=[VanillaTransformer(h=horizonte,
                                input_size=horizonte*config['input_size'],
                                hidden_size=config['neurons'],
                                conv_hidden_size=config['conv_size'],
                                n_head=config['n_heads'],
                                # Metricas de Evaluación
                                loss=metric_map[config['metric']],
                                valid_loss=metric_map[config['metric']],
                                # Escalamiento de Datos / Data Scaling
                                scaler_type='robust',
                                learning_rate=1e-3,
                                max_steps=config['max_steps'],
                                val_check_steps=50,
                                early_stop_patience_steps=-1,
                                start_padding_enabled=True),
        ],
        freq=config['freq']
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['Transformer_v2'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
    Y_hat_df['Transformer_og']  = (Y_hat_df['Transformer_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('DeepAr_og', 'y'), axis=1)
    # MAPE
    total_mape = comparativa['mape'].mean()    
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['Transformer_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['Transformer_og'])

    if config['metric'] == 'MAPE':
        tune.report({"rmse": total_mape})
    else:
        tune.report({"rmse": rmse})

# NHITS
def obj_nhits(config=None, data=None):
        
    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')        
    
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Para desescalar después:
    #original = scaler.inverse_transform(data_scaled)
    x_train['y_scaled'] = data_scaled
    horizonte = config['h']
    nf = NeuralForecast(
        models=[NHITS(h=horizonte,
                    input_size=horizonte*config['input_size'],
                    #hidden_size=config['neurons'],
                    n_freq_downsample=config['n_freq_downsample'],
                    n_pool_kernel_size = config['n_pool_kernel_size'],
                    n_blocks=[1,1,1],
                    mlp_units=3*[[config['neurons'], config['neurons']]],
                    # Metricas
                    loss=metric_map[config['metric']],
                    valid_loss=metric_map[config['metric']],
                    # Escala
                    scaler_type='robust',
                    learning_rate=config['learning_rate'],
                    max_steps=config['max_steps'],
                    val_check_steps=50,
                    early_stop_patience_steps=-1,
                    enable_progress_bar=False,
                    start_padding_enabled=True
                    ),
        ],
        freq=config['freq']
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['nhits_v2'] = scaler.inverse_transform([Y_hat_df['NHITS'].values])[0]
    Y_hat_df['nhits_og']  = (Y_hat_df['nhits_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['nhits_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['nhits_og'])
    # MAPE
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('nhits_og', 'y'), axis=1)

    total_mape = comparativa['mape'].mean()
    if config['metric'] == 'MAPE':
        tune.report({"rmse": total_mape})
    else:
        tune.report({"rmse": mae})
# Candidatos a ser Agregados. #NBEATS no lo sé. En teoría, NHITs es el sucesor.

# Posiblemente ~ Informer

# Y Patch TFT

# Temporal Fusion Transformer

# DVAE




