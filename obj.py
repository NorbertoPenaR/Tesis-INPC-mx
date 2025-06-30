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
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE
import numpy as np
import tensorflow as tf
from neuralforecast.auto import AutoNHITS
# Clasical Models



## Holt Winters
holt_param_space = {
    "trend_type": tune.choice(["mul", "add"]),
    "seasonal_type": tune.choice(["mul", "add"]),
    "damped_trend": tune.choice([True, False]),
    "use_boxcox": tune.choice([True, False]),
    "m_train": tune.randint(12, 24),
    "m_val": tune.randint(1, 3),
    "seasonal_periods": tune.grid_search([24, 24*2, 24*3]),
    #"seasonal_periods": tune.randint([24, 24*3]),
    #"" :tune.grid_search([32, 64, 128]),
}

def obj_holt_winters(config=None, data=None):

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
    #Train = x_train.copy()
    #Train['ds'] = pd.to_datetime(Train['ds'])
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
    x_train, x_val = utilities.split_data_val(data=data,
                                              train_years=config['years'],
                                              months_val=config['months'],
                                              date='ds')
    
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()

    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
    data = utilities.features_from_date(data, ds='ds')
    x_train, x_val = utilities.split_data_val(data=data,
                                             train_years=config['years'],
                                             months_val=config['months'],
                                             date='ds')
    
    features = ['quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
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
        early_stopping_rounds=150,
        verbose_eval=False,
        evals=[(dval, 'val')])

    # Evaluate model on test set and return score
    x_val['yhat'] = xgb_model_train.predict(dval)

    # MAE
    mae = mean_absolute_error(x_val['y'], x_val['yhat'])
    # RMSE
    rmse = root_mean_squared_error(x_val['y'], x_val['yhat'])
    # Send the current training result back to Tune
    tune.report({"rmse": rmse})

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
    
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()

    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = data_scaled
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

    nf.fit(df=x_train, target_col='y_scaled')

    Y_hat_df = nf.predict()
    #Y_hat_df['ds'] = Y_hat_df['ds'].apply(utilities.align_to_semi_monthly)
    Y_hat_df['rnn_v2'] = scaler.inverse_transform([Y_hat_df['RNN'].values])[0]
    Y_hat_df['rnn_og']  = (Y_hat_df['rnn_v2']*sigma)+mu
    results = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    # MAE
    mae = mean_absolute_error(Y_hat_df['y'], Y_hat_df['yhat'])
    # RMSE
    rmse = root_mean_squared_error(Y_hat_df['y'], Y_hat_df['yhat'])
    # Send the current training result back to Tune
    tune.report({"rmse": rmse})

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

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['lstm_v2'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    Y_hat_df['lstm_og']  = (Y_hat_df['lstm_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    print(comparativa)
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['lstm_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['lstm_og'])
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
    
    mu = x_train['y'].mean()
    sigma = x_train['y'].std()
    x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Para desescalar después:
    #original = scaler.inverse_transform(data_scaled)
    x_train['y_scaled'] = data_scaled
    horizonte = config['h']
    # Historic Quizá
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
    Y_hat_df['DeepAr_v2'] = scaler.inverse_transform([Y_hat_df['DeepAR'].values])[0]
    Y_hat_df['DeepAr_og']  = (Y_hat_df['DeepAr_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    print(comparativa)
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['DeepAr_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['DeepAr_og'])
    tune.report({"rmse": rmse})

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
                                loss=MAE(),
                                valid_loss=RMSE(),
                                scaler_type='robust',
                                learning_rate=1e-3,
                                max_steps=config['max_steps'],
                                val_check_steps=50,
                                early_stop_patience_steps=-1),
        ],
        freq='W-mon'
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['Transformer_v2'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
    Y_hat_df['Transformer_og']  = (Y_hat_df['Transformer_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    print(comparativa)
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['Transformer_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['Transformer_og'])
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
                    loss=MAE(),
                    valid_loss=RMSE(),
                    scaler_type='robust',
                    learning_rate=config['learning_rate'],
                    max_steps=config['max_steps'],
                    val_check_steps=50,
                    early_stop_patience_steps=-1,
                    enable_progress_bar=False,
                    ),
        ],
        freq='W-mon'
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['nhits_v2'] = scaler.inverse_transform([Y_hat_df['NHITS'].values])[0]
    Y_hat_df['nhits_og']  = (Y_hat_df['nhits_v2']*sigma)+mu
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    print(comparativa)
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['nhits_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['nhits_og'])
    tune.report({"rmse": rmse})

# Candidatos a ser Agregados. #NBEATS no lo sé. En teoría, NHITs es el sucesor.

# Posiblemente ~ Informer

# Y Patch TFT

# Temporal Fusion Transformer

# DVAE




