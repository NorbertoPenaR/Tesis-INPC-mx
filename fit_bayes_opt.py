# fit_bayes_opt.py
# Copyright (c) 2024 Norberto P. R. – All rights reserved.
# Licensed for private use only.
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from utiles import utilities
#from ray.tune.schedulers import ASHAScheduler
import obj
from functools import partial
#from ray import tune
#from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE, MSE
#from ray.tune.search import ConcurrencyLimiter
from functools import partial
from bayes_opt import BayesianOptimization

import os
import obj_bayes

# Silenciar el logger de ray.tune
import logging
metric_map = {
    'MAE': 1,
    'MAPE': 0,
    'RMSE': 2,
    'MSE': 3
}
freq_map = {
    'W-mon': 0,
    'ME': 1,
    'B':2
}
transformaciones_map = {
    'diff': 0,
    'diff_logp1': 1,
    'pct':2,
    'logp1':3,
    'none':4,
    'diff2':5
}

def fit_avg(data=None,
            horizon=None,
            Mes_val=None,
            transf=None, 
            signals=None,
            cutoff_date=None, 
            iteraciones=None
            ):
    
    avg_param_space = {
        # Temporal 
        'years':(7, 20),
        'months':(Mes_val,Mes_val)
    }

    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    
    avg_partial = partial(obj_bayes.obj_avg, data=data)

    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=avg_partial,
                                    pbounds=avg_param_space,
                                    random_state=119,
                                    verbose=2,
                                    allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']
    

#def fit_benckmarks(data=None, 
#                   horion=None,
#                   Mes_val=None,
#                   ):
    
def fit_fft(data=None, cutoff_date=None, iteraciones=None, freak=None,
            Metric=None, horizon=None, Mes_val=None, transf=None, signals=None):
    
    fft_param_space = {
        # Temporal 
        'years':(7, 20),
        'months':(Mes_val,Mes_val),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]),
        # Future Steps 
        'h':(horizon,horizon),
        # Signals
        'signals':(1, 30),
        # Frequency 
        'freq': (freq_map[freak], freq_map[freak]), # 0 'W-mon', 1 'ME'.
    }

    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    
    fft_partial = partial(obj_bayes.obj_fft, data=data)

    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=fft_partial,
                                    pbounds=fft_param_space,
                                    random_state=119,
                                    verbose=2,
                                    allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# Holt Winters
def fit_holt_winters(data=None, cutoff_date=None, iteraciones=None, freak=None,
                    Metric=None, horizon=None, Mes_val=None, transf=None):

    holt_param_space = {
        # Holt Parameters / Binary 
        "trend_type":(0,1),
        "seasonal_type":(0,1),
        "damped_trend": (0,1),
        "use_boxcox":(0,1),
        # Data Splitting Parameters / Continuous
        'years':(7, 30),
        'months':(Mes_val,Mes_val),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]), 
        # Future Steps 
        'h':(horizon,horizon),
        "seasonal_periods": (94, 104),
        # The right seasonal period is achieved by applying 
        # the Discrete Fourier Transformation to our
        # Target Variable.
        # Frequency 
        'freq': (freq_map[freak], freq_map[freak]), # 0 'W-mon', 1 'ME'.
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    holt_w_partial = partial(obj_bayes.obj_holt_winters, data=data)

    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=holt_w_partial,
                                    pbounds=holt_param_space,
                                    random_state=119,
                                    verbose=2,
                                    allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=25)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# XGBoost
def fit_xgb(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    
    xgb_params = {
        'years':(15, 30),
        'months':(Mes_val, Mes_val),
        # XGB Params
        'max_depth':(2, 15),
        'colsample_bytree':(.5,.9),
        'subsample':(.6,.95),
        'alpha':(0,5),
        'eta':(.3,.6),
        'lambdaa':(0,5),
        'num_boost_round':(50, 150),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Signals
        'signals':(signals, signals),
        # Future Steps
        'h':(horizon, horizon),
        # Features
        'feats':(feats,feats),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf])
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    #if len(data)>52:
    xgb_partial = partial(obj_bayes.obj_xgb, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=xgb_partial,
                                        pbounds=xgb_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=True)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=10, n_iter=50)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']
    
# RNN
def fit_rnn(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    # NO TOCAR 
    rnn_params = {
        # Data Splitting Params
        'years':(5, 12),
        'months':(Mes_val,Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Neural Network Parameters
        'input_size':(6,9),
        'neurons':(3,4),
        'layers':(1,2),
        "max_steps": (10, 70),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Features
        'feats':(feats,feats),
        # Signals
        'signals':(signals, signals),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    rnn_partial = partial(obj_bayes.obj_rnn, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=rnn_partial,
                                        pbounds=rnn_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=3, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# LSTM
def fit_lstm(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):

    lstm_params = {
        # Data Splitting Params
        'years':(5, 12),
        'months':(Mes_val,Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Neural Network Parameters
        'input_size':(6,10), # No meter menos 
        'layers':(1,3), # 1 y 2 funconan bien con 0 y 1 features
        "max_steps": (10, 100), # NO CAMBIAR
        'neurons':(2, 5), # NO CAMBIAR  (3,4) funciona bien con 0 y 1 features
        #'learning_rate': (0.001, 0.001), # (0.001, 0.01)
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Signals
        'signals':(signals, signals),
        # Features
        'feats':(feats,feats),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    
    lstm_partial = partial(obj_bayes.obj_lstm, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=lstm_partial,
                                    pbounds=lstm_params,
                                    random_state=119,
                                    verbose=2,
                                    allow_duplicate_points=False)
    
    # Perform Bayesian optimization
    optimizer.maximize(init_points=3, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# Deep Ar
def fit_deep_ar(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
                signals=None):

    deep_ar_params = {
        # Data Splitting Params.
        'years':(8, 15),
        'months':(Mes_val,Mes_val),
        # Implica tener un horizonte mayor a 2 y 4 meses.
        # Es decir, Dado el Máximo Rango
        # Tomar ese como punto de partida para el resto de experimentos que se harán. 
        # Entonces, se requieren 4*4=16 Es decir, para evaluar en datos de test
        # Se necesitarán 16+4 = 20 
        # 20*2 = 40; El doble.
        # Future Steps
        'h':(horizon, horizon),
        # Neural Network Params
        'input_size':(2, 5),
        'layers':(1,4),
        'trajectories':(100, 100),
        'learning_rate':(0.01, 0.01),#(0.001, 0.01),#(1e-4, 1e-1),
        "max_steps": (25, 100),
        'neurons':(2,6),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Signals
        'signals':(signals, signals),
        # Features
        'feats':(feats,feats),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    deepAr_partial = partial(obj_bayes.obj_deep_ar, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=deepAr_partial,
                                    pbounds=deep_ar_params,
                                    random_state=119,
                                    verbose=2,
                                    allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# Transformers
# De acuerdo con un paper, lo ideal es comenzar con poco
# Encontrar el sweet spot. Entre más complejo sea un modelo
# La probabilidad de que este se overfitee es mayor. 
# Haremos dos experimentos. 
# Uno consistirá en incluir las variables exogenas temporales, 
# senoidales y el componente de la señal

def fit_transformer(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                    Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
                    signals=None):

    transformer_params = {
        'years':(4, 15),
        'months':(Mes_val, Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Params
        'input_size':(2,6),
        'neurons':(2,6),
        'conv_size':(2,2),
        'n_heads':(2, 4), # Using more than 10 heads is pointless, given the fact that 2 heads were better than 14 heads. 
        "max_steps": (25, 125),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Features
        'feats':(feats,feats),
        # Learning Rate
        'learning_rate':(.0001,.0001),
        # Fourier
        'signals':(signals, signals),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    transformer_partial = partial(obj_bayes.obj_transformer, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=transformer_partial,
                                        pbounds=transformer_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']
 
# NHITS
def fit_nhits(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):

    nhits_params={
        # Split Data Params
        'years':(3, 8),
        'months':(Mes_val,Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Neural Network Parameters
        'input_size':(1, 6),
        'neurons':(7,10),
        "max_steps": (500, 2500),
        #"n_pool_kernel_size": tune.choice([3 * [2], 3 * [4], 3 * [8], [8, 4, 1], [16, 8, 1]]),
        #"n_freq_downsample": tune.choice([[168, 24, 1],
        #                                [24, 12, 1],
        #                                [180, 60, 1],
        #                                [60, 8, 1],
        #                                [40, 20, 1]]),
        "learning_rate": (1e-4, 1e-1),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Signals
        'signals':(signals, signals),
        # Features
        'feats':(feats,feats),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    nhits_partial = partial(obj, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=nhits_partial,
                                    pbounds=nhits_params,
                                    random_state=42,
                                    verbose=2,
                                    allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# DVAE
def fit_d3vae(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    
    transformer_params = {
        'years':(4, 15),
        'months':(Mes_val, Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Params
        'input_size':(2,6),
        'neurons':(2,6),
        'conv_size':(2,2),
        'n_heads':(2, 4), # Using more than 10 heads is pointless, given the fact that 2 heads were better than 14 heads. 
        "max_steps": (25, 125),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Features
        'feats':(feats,feats),
        # Learning Rate
        'learning_rate':(.0001,.0001),
        # Fourier
        'signals':(signals, signals),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    transformer_partial = partial(obj_bayes.obj_transformer, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=transformer_partial,
                                        pbounds=transformer_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']