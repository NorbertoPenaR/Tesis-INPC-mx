# Cross Validation
# Statistical Models

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
import obj_cv_bayes

simulation_dates = utilities.ultimos_dias_meses(n=6, frecuencia=3, referencia='2025-01-01')


from time import time

metric_map = {
    'MAPE': 0,
    'MAE': 1,
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

# Get the current time before forecasting starts, this will be used to measure the execution time
#init = time()

# Get the current time after the forecasting ends
#end = time()

# Calculate and print the total time taken for the forecasting in minutes
#print(f'Forecast Minutes: {(end - init) / 60}')

# FFT
def fit_fft_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    fft_params={
        'years':10,
        'months':3,
        'h':52,
    }

# Naive_Avg_Random_Walk
def fit_avg_rwd_naive_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
        Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
        signals=None):
    
    classic_params = {
        'years':(5,25),
        'months':(3,4),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]), 
        'h':(horizon,horizon),
        'freq':(freq_map[freak], freq_map[freak]), # 0 'W-mon', 1 'ME'.
    }
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    avg_rwd_naive_partial = partial(obj_cv_bayes.obj_avg_rwd_naive_cv, data=data)

    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=avg_rwd_naive_partial,
                                    pbounds=classic_params,
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
def fit_holt_winters_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    
    holt_param_space = {
        # Holt Parameters / Binary 
        "trend_type":(0,1),
        "seasonal_type":(0,1),
        "damped_trend": (0,1),
        "use_boxcox":(0,1),
        # Data Splitting Parameters / Continuous
        'years':(12, 22),
        'months':(Mes_val,Mes_val),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]), 
        # Future Steps 
        'h':(horizon,horizon),
        "seasonal_periods": (99, 99),
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

    holt_w_partial = partial(obj_cv_bayes.obj_holt_winters_cv, data=data)

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
def fit_xgb_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    
    xgb_params = {
        'years':(10, 25),
        'months':(Mes_val, Mes_val),
        # XGB Params
        'max_depth':(5, 15),
        'colsample_bytree':(.5,.8),
        'subsample':(.5,.8),
        'alpha':(0,5),
        'eta':(.3,.3),
        'lambdaa':(0,5),
        'num_boost_round':(50, 300),
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
                  transformaciones_map[transf]),
        'input_mult':(2,5)
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    #if len(data)>52:
    xgb_partial = partial(obj_cv_bayes.obj_xgb_cv, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=xgb_partial,
                                        pbounds=xgb_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=True)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# RNN
def fit_rnn_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):
    # NO TOCAR 
    rnn_params = {
        # Data Splitting Params
        'years':(10, 20),
        'months':(Mes_val,Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Neural Network Parameters
        'input_size':(6,9), #Tesis Originales
        #'input_size':(1,2), # venta Mex
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
                  transformaciones_map[transf]),
        'batch_size':(6, 8) #Tesis
        #'batch_size':(2, 2) # Venta Mabe
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    rnn_partial = partial(obj_cv_bayes.obj_rnn_cv, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=rnn_partial,
                                        pbounds=rnn_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# LSTM
def fit_lstm_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
            signals=None):

    lstm_params = {
        # Data Splitting Params
        'years':(8, 18),
        'months':(Mes_val,Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Neural Network Parameters
        'input_size':(3,6), # No meter menos 
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
                  transformaciones_map[transf]),
        # Batch Size
        'batch_size':(6, 8)
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    lstm_partial = partial(obj_cv_bayes.obj_lstm_cv, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=lstm_partial,
                                    pbounds=lstm_params,
                                    random_state=119,
                                    verbose=2,
                                    allow_duplicate_points=False)
    
    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']

# Deep Ar
def fit_deep_ar_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
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
        'layers':(1,2), # con una capa salen bien, más cuando hay senoidales. 
        # Y cuando se usaron 4 capas, no es como que haya resultado bien. 
        # Salio una predicción Lineal. Lo mejor sería evitarlas. 
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
                  transformaciones_map[transf]),
        # Batch Size
        'batch_size':(6, 8)
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    deepAr_partial = partial(obj_cv_bayes.obj_deep_ar_cv, data=data)
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

def fit_transformer_cv(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                    Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
                    signals=None):

    transformer_params = {
        'years':(10, 20),
        'months':(Mes_val, Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Params
        'input_size':(2,6),
        'neurons':(3,7),
        'conv_size':(2,2),
        'n_heads':(2, 5), # Using more than 10 heads is pointless, given the fact that 2 heads were better than 14 heads. 
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
                  transformaciones_map[transf]),
        # Batch Size
        'batch_size':(7, 7)
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    transformer_partial = partial(obj_cv_bayes.obj_transformer_cv, data=data)
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

def fit_patch_cv(data=None, cutoff_date=None, iteraciones=None, freak=None,
                Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
                signals=None):
    
    print('Patch - Transformer')
    

def fit_dvae_cv(data=None, cutoff_date=None, iteraciones=None, freak=None,
                Metric=None, horizon=None, Mes_val=None, feats=None, transf=None,
                signals=None):
    
    dvae_params = {
        'years':(12, 20),
        'months':(Mes_val, Mes_val),
        # Future Steps
        'h':(horizon, horizon),
        # Params
        'input_size':(1,4),
        'neurons':(4,8), #
        'layers':(1,2),
        #'batch_size':(4, 8), # cantidad d sequencias a ser procesadas en paralelo.
        "max_steps": (50, 125),
        'batch_size':(5, 5),
        'dropout':(0, 0),
        'beta_kl':(.2, .5),
        'teacher_forcing':(.5, .7),
        'dimension':(2,5),
        # Frequency
        'freq': (freq_map[freak], freq_map[freak]),
        # Metric
        'metric': (metric_map[Metric],metric_map[Metric]),
        # Features
        'feats':(feats,feats),
        # Learning Rate
        #'learning_rate':(.0001,.0001),
        # Fourier
        'signals':(signals, signals),
        # Trasnformation
        'transf':(transformaciones_map[transf],
                  transformaciones_map[transf]) 
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    transformer_partial = partial(obj_cv_bayes.obj_dvae_cv, data=data)
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=transformer_partial,
                                        pbounds=dvae_params,
                                        random_state=119,
                                        verbose=2,
                                        allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=iteraciones)
    dic_params = dict()
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    return dic_params, dic_params['accuracy']