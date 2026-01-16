# obj_cv_bayes.py
# Copyright (c) 2024 Norberto P. R. – All rights reserved.
# Licensed for private use only.

from sklearn.preprocessing import MinMaxScaler
## 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from utiles import utilities
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelos Clasicos, BenchMarks. 
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage, Naive, RandomWalkWithDrift

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, VanillaTransformer, LSTM, RNN, NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE, MSE
import numpy as np
import tensorflow as tf
from neuralforecast.auto import AutoNHITS
# 
import accuracy

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean
from mlforecast.target_transforms import Differences
from xgboost import XGBRegressor

# Clasical Models
import os
import pytorch_lightning as pl
trainer = pl.Trainer(logger=False, enable_progress_bar=False)
import yaml
from dvae_v2.train import main as train_main
from dvae_v2.predict import main as predict_main

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
metric_map_number_fun = {
    0: MAPE(),
    1: MAE(),
    2: RMSE(),
    3: MSE()
}

metric_map = {
    0: MAPE(),
    1: MAE(),
    2: RMSE(),
    3: MSE()
}

frequency_map = {
    0:'W-mon',
    1:'ME',
    2:'B'
}

transformaciones_map = {
    'diff': 0,
    'diff_logp1': 1,
    'pct':2,
    'logp1':3,
    'none':4
}

inverse_map_tr = {
    0:'diff',
    1:'diff_logp1',
    2:'pct',
    3:'logp1',
    4:'none',
    5:'diff2'
}
simulation_dates = utilities.ultimos_dias_meses(n=6, frecuencia=3, referencia='2025-01-01')
import process_data
inpc_path_Q = 'ca56_2018a.csv'
inpc_Q = process_data.limpiar_csv_inegi(inpc_path_Q)
weekly_inpc = process_data.inpc_data_weekly(datos=inpc_Q)

inpc_path_M = 'ca55_2018a.csv'
inpc_M = process_data.limpiar_csv_inegi(inpc_path_M)
monthly_inpc = process_data.inpc_monthly(datos=inpc_M)

# NAIVE
def naive_improvement_penalty(y_true, y_last, y_pred ,s_recent=None, gamma=0.5):
    """Penaliza si NO mejoras al naive."""
    H = len(y_true)
    h = np.arange(1, H+1)
    if s_recent is None: s_recent = 0.0
    y_naive = y_last + s_recent*h
    imp = mean_absolute_error(y_true, y_naive) - mean_absolute_error(y_true, y_pred)  # >0: mejoras
    return float(gamma * max(0.0, -imp))  # penaliza solo si no mejoras

# Avg_rwd_naive
def obj_avg_rwd_naive_cv(data=None,
                        transf =None,
                        years=None,
                        months=None,
                        metric=None,
                        h=None,
                        freq=None):
    
    config = {'years':years,
            'months':months,
            'metric':metric,
            'h':int(h),
            'freq':freq,
            'transf':transf}
    
    data['ds'] = pd.to_datetime(data['ds'])

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    x_train_val = pd.concat([x_train, x_val])

    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, 
                                    frecuencia=12, 
                                    referencia=x_train_val.ds.max())

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []
    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]
        #whole = pd.concat([x_train, x_val])
        
        avg_method = HistoricAverage()
        naive_method = Naive()
        drift_method = RandomWalkWithDrift()
        sf = StatsForecast(models=[drift_method, avg_method, naive_method], 
                           freq=frequency_map[config['freq']],)

        sf.fit(x_train)
        horizonte= int(config['h'])
        fcasts = sf.forecast(df=x_train, 
                             h=horizonte, 
                             level=[95])
        comparativa = fcasts.merge(x_val, on=['unique_id', 'ds'], how='inner')

        #print(comparativa)
            
        # MAE
        mae = mean_absolute_error(comparativa['y'], comparativa['HistoricAverage'])
        cv_mae.append(mae)
        # RMSE
        rmse = root_mean_squared_error(comparativa['y'], comparativa['HistoricAverage'])
        cv_rmse.append(rmse)
        # MAPE
        comparativa['mape'] = comparativa.apply(accuracy.mape, args=('HistoricAverage', 'y'), axis=1)
        total_mape = comparativa['mape'].mean()
        cv_mape.append(total_mape)
        # MSE
        mse = mean_squared_error(comparativa['y'], comparativa['HistoricAverage'])
        cv_mse.append(mse)
    
    #print('Salio')
    #print(cv_mae)
    if config['metric'] == 0:
       return -round(np.mean(cv_mape) , 3)

    elif int(config['metric']) == 1: 
        #print('MAE')
        return -round(np.mean(cv_mae) , 3)
        
    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)
    
    #print('Q paso?')

# Holt Winters
def obj_holt_winters_cv(data=None,
                    transf =None,
                    trend_type=None,
                    seasonal_type=None,
                    damped_trend=None,
                    use_boxcox=None,
                    years=None,
                    months=None,
                    metric=None,
                    h=None,
                    seasonal_periods=None,
                    freq=None):
    
    config = {'trend_type':trend_type,
            'seasonal_type':seasonal_type,
            'damped_trend':damped_trend,
            'use_boxcox':use_boxcox,
            'years':years,
            'months':months,
            'metric':metric,
            'h':int(h),
            'seasonal_periods':seasonal_periods,
            'freq':freq,
            'transf':transf}
    
    # Holt-Winters Parameters
    if config['trend_type'] >= float(.5):
        config['trend_type'] = 'mul'
    elif config['trend_type'] < .5:
        config['trend_type'] = 'add'

    if config['seasonal_type'] >= .5:
        config['seasonal_type'] = 'mul'
    elif config['seasonal_type'] < .5:
        config['seasonal_type'] = 'add'

    if config['damped_trend'] >= .5:
        config['damped_trend'] = True
    elif config['damped_trend'] <.5:
        config['damped_trend'] = False

    if config['use_boxcox'] >= .5:
        config['use_boxcox'] = True
    elif config['use_boxcox'] < .5:
        config['use_boxcox'] = False

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    x_train_val = pd.concat([x_train, x_val])

    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []
    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]

        transf = config['transf']
        if config['transf'] == 0:
            x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
            x_train = x_train.dropna()  #El primer valor será NaN

        elif config['transf'] == 1:
            x_train['y_log'] = np.log1p(x_train['y'])
            x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
            x_train = x_train.dropna()

        elif config['transf'] == 2:
            x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
            x_train = x_train.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3:
            x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

        elif config['transf'] == 4:
            x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

        elif config['transf'] == 5:
            x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
            x_train = x_train.dropna()

        # id
        id_holt = list(x_train.unique_id.unique())[0]

        # Estandarización
        transf = config['transf']
        mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
        sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
        x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu) / sigma

        # Escalar
        scaler = MinMaxScaler(feature_range=(1, 10))
        x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

        HoltWinters = ExponentialSmoothing(x_train['y_scaled'],
                        dates=x_train['ds'],
                        trend=config['trend_type'],  # 'add'
                        seasonal=config['seasonal_type'],  # 'add', 'mul'
                        seasonal_periods= int(config['seasonal_periods']),#int(info_frec[1]['periodo']),
                        damped_trend=config['damped_trend'],  # True / False
                        use_boxcox=config['use_boxcox'],
                        freq=frequency_map[config['freq']]
                        ).fit()
        
        H = pd.DataFrame()
        H['holt_w'] = HoltWinters.forecast(steps=config['h'] + len(x_val))

        # Paso 4.1: Invertir escalado
        H['holt_w_v2'] = scaler.inverse_transform(H[['holt_w']])

        # Paso 4.2: Invertir estandarización
        H['Holt'] = H['holt_w_v2'] * sigma + mu

        # Paso 4.3: Reconstruir serie original desde el último valor real
        transf = config['transf']
        if transf == 0:
            H['HoltW_og'] = utilities.reconstruccion_diff(H['Holt'], x_train['y'].iloc[-1])
        elif transf == 1:
            H['HoltW_og'] = utilities.reconstruccion_log_diff(H['Holt'], x_train['y'].iloc[-1])
        elif transf == 2:
            H['HoltW_og'] = utilities.reconstruccion_pct(H['Holt'], x_train['y'].iloc[-1])
        elif transf == 3:
            H['HoltW_og'] = np.expm1(H['Holt'])
        elif transf == 4:
            H['HoltW_og'] = H['Holt']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            H['HoltW_og'] = utilities.reconstruccion_diff2(H['Holt'], y_t_1, y_t_2)

        H = H.reset_index().rename(columns={'index':'ds'})
        H['unique_id'] = id_holt
        x_val = x_val.merge(H, on=['ds', 'unique_id'], how='inner')
        #x_val['yhat_og'].fillna(0, inplace=True)
        #x_val['yhat_og'] = x_val['yhat'].clip(lower=0)
        #print(x_val)
        #Val['yhat'] = Val['yhat'].astype(int)
        try:
            # MAPE
            x_val['mape'] = x_val.apply(accuracy.mape, args=('HoltW_og','y'), axis=1)
            mape_total = x_val['mape'].mean()
            cv_mape.append(mape_total)

            # MAE
            mae = mean_absolute_error(x_val['y'], x_val['HoltW_og'])
            cv_mae.append(mae)

            # RMSE
            rmse = root_mean_squared_error(x_val['y'], x_val['HoltW_og'])
            cv_rmse.append(rmse)

            # MSE
            mse = mean_squared_error(x_val['y'], x_val['HoltW_og'])
            cv_mse.append(mse)


            # With Bayesian optimization we return the error, but since we maximizing
            # we have to send a negative version of it. 
        except Exception as e:
                return -10000
    if config['metric'] == 0:
        return -round(np.mean(cv_mape) , 3)
        
    elif config['metric'] == 1:
        #print('MAE')
        return -round(np.mean(cv_mae) , 3)

    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)

# XGB
def obj_xgb_cv(years=None,
                months=None,
                # XGB params
                max_depth=None,
                colsample_bytree=None,
                subsample=None,
                alpha=None,
                eta=None,
                lambdaa=None,
                num_boost_round=None,
                metric=None,
                # Senoidales
                signals=None,
                # Horizon
                h=None,
                # Frequency
                freq=None,
                # Data
                data=None,
                #Features,
                feats=None,
                # Transformaciones
                transf=None,
                input_mult=None):
    
    try:
        
        config = {
            'years':int(years),
            'months':int(months),
            'max_depth':int(max_depth),
            'colsample_bytree':colsample_bytree,
            'subsample':subsample,
            'alpha':alpha,
            'eta':eta,
            'lambdaa':lambdaa,
            'num_boost_round':int(num_boost_round),
            'metric':metric,
            'signals':int(signals),
            'h':int(h),
            'freq':freq,
            'feats':int(feats),
            # Transformaciones
            'transf':transf,
            'input_mult':input_mult
        }

        # Rango
        cyclic_cols = {
                'month': 12,
                'weekofyear': 52,
                'dayofyear': 366,
                'dayofmonth': 31,
                'quarter': 4,
                }
        
        # Variables Temporales
        data = utilities.features_from_date(data, 'ds')
        for col, max_val in cyclic_cols.items():
            data = utilities.add_cyclic_features(data, col, max_val)

        # División de Conjuntos.
        x_train, x_val = utilities.split_data_val(data=data, 
                                                train_years=int(config['years']), 
                                                months_val=int(config['months']), 
                                                date='ds')
        
        x_train_val = pd.concat([x_train, x_val])
        
        simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())

        cv_i = 0
        cv_mae = []
        cv_rmse = []
        cv_mape = []
        cv_mse = []
        for dt_cv in simulation_dates_cv[:-1]:
            cv_i = cv_i + 1 
            x_train = x_train_val[x_train_val['ds']<dt_cv]
            x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]


            xgb_exo = MLForecast(
                models=XGBRegressor(
                    n_estimators=config['num_boost_round'], # 200 seems to work fine
                    max_depth=config['max_depth'],
                    learning_rate=config['eta'],
                    subsample=config['subsample'],
                    colsample_bytree=config['colsample_bytree'],
                    random_state=119,
                    reg_alpha=config['alpha'],
                    reg_lambda=config['lambdaa']
                ),
                freq=frequency_map[config['freq']],
                lags=list(range(1,int(52*config['input_mult']+1))),
                target_transforms=[Differences([1])],
                #target_transforms=[GlobalSklearnTransformer(sk_log1p), Differences([1])],
            )
        
            # XGBoost only has 2 options, unless lags are added.
            if config['feats']==0:
                xgb_exo.fit(x_train[['ds', 'unique_id', 'y']])
                dff = xgb_exo.predict(h=int(config['h']))
                
            elif config['feats']==1:
                #feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
                feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
                #feats = feats 
                #print(x_train[['ds', 'unique_id', 'y']+feats])
                xgb_exo.fit(x_train[['ds', 'unique_id', 'y']+feats], static_features=[])

                future_df = xgb_exo.make_future_dataframe(h=int(config['h']))
                future_df = utilities.features_from_date(future_df, 'ds')
                for col, max_val in cyclic_cols.items():
                    future_df = utilities.add_cyclic_features(future_df, col, max_val)
                
                dff = xgb_exo.predict(h=int(config['h']), X_df=future_df)
            

            elif config['feats']==2:# Features Senoidales added.
                # Senoidales
                senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y'],
                                                    top_k=int(config['signals']),
                                                    extra_steps=int(config['h']))
                # Entrenamiento
                for c in senoidales.columns:
                    x_train[c] = (senoidales[c]).values[:len(x_train)]

                feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos',
                                                                                    'sin']]
                feats += list(senoidales.columns)[:-1]
                xgb_exo.fit(x_train[['ds', 'unique_id', 'y']+feats], static_features=[])                
                future_df = xgb_exo.make_future_dataframe(h=int(config['h']))
                future_df = utilities.features_from_date(future_df, 'ds')

                for col, max_val in cyclic_cols.items(): # Temporales
                    future_df = utilities.add_cyclic_features(future_df, col, max_val)
                for c in senoidales.columns: # Fast Fourier Transform 
                    #x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))]
                    future_df[c] = (senoidales[c]).values[len(x_train):]

                dff = xgb_exo.predict(h=int(config['h']), X_df=future_df)
            
            comparativa = dff.merge(x_val, on=['unique_id', 'ds'], how='inner')
            #print(comparativa)
            #input()

            # MAE
            mae = mean_absolute_error(comparativa['y'], comparativa['XGBRegressor'])
            cv_mae.append(mae)
            # RMSE
            rmse = root_mean_squared_error(comparativa['y'], comparativa['XGBRegressor'])
            cv_rmse.append(rmse)
            # MAPE
            comparativa['mape'] = comparativa.apply(accuracy.mape, args=('XGBRegressor', 'y'), axis=1)
            total_mape = comparativa['mape'].mean()
            cv_mape.append(total_mape)
            # MSE
            mse = mean_squared_error(comparativa['y'], comparativa['XGBRegressor'])
            cv_mse.append(mse)

        if config['metric'] == 0:
            return -round(np.mean(cv_mape) , 3)

        elif config['metric'] == 1: 
            return -round(np.mean(cv_mae) , 3)
            
        elif config['metric'] == 2:
            return -round(np.mean(cv_rmse) , 3)

        elif config['metric'] == 3:
            return -round(np.mean(cv_mse) , 3)
    
    except Exception as e:
        print(e)
        return -1000   
    
    # Evaluate model on test set and return score
    #x_val['yhat'] = xgb_model_train.predict(dval)
    #transf = config['transf']
    #if transf == 0:
    #    x_val['yhat_og'] = utilities.reconstruccion_diff(x_val['yhat'], x_train['y'].iloc[-1])

# RNN
def obj_rnn_cv(data=None,
            # Parametros de Corte Temporal
            years=None,
            months=None,
            # Horizonte
            h=None,
            # RNN Parameters
            input_size=None,
            neurons=None,
            layers=None,
            max_steps=None,
            batch_size=None,
            # Frequencia
            freq=None,
            # Metrica
            metric=None,
            #Features
            feats=None,
            # Transformaciones
            transf=None,
            signals=None
            ):

    config = {'years':years,
        'months':months,
        'h':int(h),
        'input_size':int(input_size),
        'neurons':neurons,
        'layers':layers,
        'max_steps':max_steps,
        'freq':freq,
        'metric':metric,
        'feats':int(feats),
        'signals': int(signals),
        'batch_size': int(batch_size), # 2^5 = 32, 2^6 = 64, 2^7 = 128, 2^8 = 256
        'transf':transf
    }

    # Rango
    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    # Variables Temporales
    data = utilities.features_from_date(data, 'ds')
    # Transformación Senoidal d Variables Temporales
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)
    
    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    x_train_val = pd.concat([x_train, x_val])
    
    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size'])

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []
    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]

        # Transformaciones
        if config['transf'] == 0:
            x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
            x_train = x_train.dropna()  #El primer valor será NaN

        elif config['transf'] == 1:
            x_train['y_log'] = np.log1p(x_train['y'])
            x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
            x_train = x_train.dropna()

        elif config['transf'] == 2:
            x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
            x_train = x_train.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3:
            x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

        elif config['transf'] == 4:
            x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

        elif config['transf'] == 5:
            x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
            x_train = x_train.dropna()
        
        # Estandarización
        transf = config['transf']
        mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
        sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
        x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu) / sigma
    
        # Escalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
        # Senoidales
        senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'], 
                                                                top_k=config['signals'], 
                                                                extra_steps=config['h'])
        for c in senoidales.columns:
            x_train[c] = (senoidales[c]).values[:len(x_train)]

        # Horizonte
        horizonte = config['h']

        # Variables Exogenas
        if config['feats']==0: # Ninguna
            features=None
        elif config['feats']==1: # Temporales
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        elif config['feats']==2: # Temporales + Senoidales
            # Features Senoidales added.
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
            features += list(senoidales.columns)[:-1]

        nf = NeuralForecast(
            models=[RNN(h=horizonte,
                        input_size=int(horizonte*config['input_size']),
                        # Metricas de Evaluación
                        loss=metric_map_number_fun[config['metric']],
                        valid_loss=metric_map_number_fun[config['metric']],
                        # Escalamiento de Datos & Posibles variables exogenas
                        scaler_type='standard',
                        encoder_n_layers=int(config['layers']),
                        encoder_hidden_size=config['neurons'],
                        decoder_hidden_size=config['neurons'],
                        decoder_layers=int(config['layers']),
                        max_steps=int(config['max_steps']),
                        futr_exog_list=features,
                        batch_size=config['batch_size'],
                        early_stop_patience_steps=10,
                        val_check_steps=10,
                        #stat_exog_list=['airline1'],
                        enable_progress_bar=False,
                        #start_padding_enabled=True
                        random_seed=119
                        )
            ],
            freq=frequency_map[config['freq']]
        )

        nf.fit(df=x_train, target_col='y_scaled', verbose=False, val_size=horizonte)

        # Variables Exogenas
        if config['feats']==0: # Ninguna
            Y_hat_df = nf.predict(verbose=0)
        
        elif config['feats']==1: # Temporales
            df_features = nf.make_future_dataframe()
            # Data Features
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        elif config['feats']==2: # Temporales + Senoidales
            df_features = nf.make_future_dataframe()
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            for c in senoidales.columns:
                df_features[c] = (senoidales[c]).values[len(x_train):]
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        # Se Reconstruye la Inversa. 
        Y_hat_df['RNN'] = scaler.inverse_transform([Y_hat_df['RNN'].values])[0]
        Y_hat_df['RNN'] = (Y_hat_df['RNN']*sigma) + mu

        # Reconstruccion
        transf = config['transf']
        if transf == 0:
            Y_hat_df['rnn_og'] = utilities.reconstruccion_diff(Y_hat_df['RNN'], x_train['y'].iloc[-1])
        elif transf == 1:
            Y_hat_df['rnn_og'] = utilities.reconstruccion_log_diff(Y_hat_df['RNN'], x_train['y'].iloc[-1])
        elif transf == 2:
            Y_hat_df['rnn_og'] = utilities.reconstruccion_pct(Y_hat_df['RNN'], x_train['y'].iloc[-1])
        elif transf == 3:
            Y_hat_df['rnn_og'] = np.expm1(Y_hat_df['RNN'])
        elif transf == 4:
            Y_hat_df['rnn_og'] = Y_hat_df['RNN']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            Y_hat_df['rnn_og'] = utilities.reconstruccion_diff2(Y_hat_df['RNN'], y_t_1, y_t_2)
    
        comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
        # MAE
        mae = mean_absolute_error(comparativa['y'], comparativa['rnn_og'])
        cv_mae.append(mae)
        # RMSE
        rmse = root_mean_squared_error(comparativa['y'], comparativa['rnn_og'])
        cv_rmse.append(rmse)
        # MAPE
        comparativa['mape'] = comparativa.apply(accuracy.mape, args=('rnn_og', 'y'), axis=1)
        total_mape = comparativa['mape'].mean()
        cv_mape.append(total_mape)
        # MSE
        mse = mean_squared_error(comparativa['y'], comparativa['rnn_og'])
        cv_mse.append(mse)

    if config['metric'] == 0:
        return -round(np.mean(cv_mape) , 3)

    elif config['metric'] == 1: 
        return -round(np.mean(cv_mae) , 3)
        
    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)
    
# LSTM
def obj_lstm_cv(data=None,
                h=None,
                years=None,
                months=None,
                input_size=None,
                layers=None,
                max_steps=None,
                neurons=None,
                #learning_rate=None,
                freq=None,
                metric=None,
                #Features,
                feats=None,
                # Transformaciones
                signals=None,
                transf=None,
                batch_size=None
                ):

    config={'years':int(years),
            'months':int(months),
            'h':int(h),
            'input_size':int(input_size),
            'layers':int(layers),
            'max_steps':int(max_steps),
            'neurons':neurons,
            #'learning_rate':learning_rate,
            'freq':freq,
            'metric':metric,
            'feats':int(feats),
            'signals': int(signals),
            'transf':transf,
            'batch_size': int(batch_size)
        }

    # Rango
    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    # Variables Temporales
    data = utilities.features_from_date(data, 'ds')
    # Transformación Senoidal d Variables Temporales
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    # Se junta el conjunto de validación para formar el gran conjunto, el cual será dividido en 4 splits con horizontes 
    # Previamente definidos por el usuario desde el modulo main.py 
    x_train_val = pd.concat([x_train, x_val])
    
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size']) # 64, 128 

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []
    
    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]

        # Transformaciones
        if config['transf'] == 0:
            x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
            x_train = x_train.dropna()  #El primer valor será NaN

        elif config['transf'] == 1:
            x_train['y_log'] = np.log1p(x_train['y'])
            x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
            x_train = x_train.dropna()

        elif config['transf'] == 2:
            x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
            x_train = x_train.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3:
            x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

        elif config['transf'] == 4:
            x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

        elif config['transf'] == 5:
            x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
            x_train = x_train.dropna()
        
        # Estandarización
        transf = config['transf']
        mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
        sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
        x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu) / sigma
    
        # Escalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
        # Senoidales
        senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'], 
                                                                top_k=config['signals'],
                                                                extra_steps=config['h'])
        for c in senoidales.columns:
            x_train[c] = (senoidales[c]).values[:len(x_train)]

        # Horizonte
        horizonte = config['h']

        # Variables Exogenas
        if config['feats']==0: # Ninguna
            features=None
        elif config['feats']==1: # Temporales
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        elif config['feats']==2: # Temporales + Senoidales
            # Features Senoidales added.
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
            features += list(senoidales.columns)[:-1]

        nf = NeuralForecast(
            models=[LSTM(h=horizonte,
                        input_size=horizonte*config['input_size'],
                        encoder_n_layers=int(config['layers']),
                        encoder_hidden_size=config['neurons'],
                        decoder_hidden_size=config['neurons'],
                        decoder_layers=int(config['layers']),
                        loss=metric_map[config['metric']],
                        valid_loss=metric_map[config['metric']],
                        futr_exog_list=features,
                        max_steps=int(config['max_steps']),
                        val_check_steps=5,
                        batch_size=config['batch_size'],
                        early_stop_patience_steps=15,
                        scaler_type='standard',
                        enable_progress_bar=False,
                        random_seed=119,
                        enable_checkpointing=True,
                        #logger=True, 
                        #logger=TensorBoardLogger("logs/"),
                        ),
            ],
            freq=frequency_map[config['freq']]
        )

        nf.fit(df=x_train, target_col='y_scaled', verbose=False, val_size=horizonte)
        #nf.predict_insample
        # Access logs
        #model = nf.models[0]
        #print(model)
        
        #history = model.
        #print(history)

        # Variables Exogenas    
        if config['feats']==0: # Ninguna
            Y_hat_df = nf.predict(verbose=0)
        
        elif config['feats']==1: # Temporales
            df_features = nf.make_future_dataframe()
            # Data Features
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        elif config['feats']==2: # Temporales + Senoidales
            df_features = nf.make_future_dataframe()
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            for c in senoidales.columns[:-1]:
                df_features[c] = (senoidales[c]).values[len(x_train):]
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        # Se Reconstruye la Inversa. 
        Y_hat_df['LSTM'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
        Y_hat_df['LSTM'] = (Y_hat_df['LSTM']*sigma) + mu

        # Reconstruccion
        transf = config['transf']
        if transf == 0:
            Y_hat_df['lstm_og'] = utilities.reconstruccion_diff(Y_hat_df['LSTM'], x_train['y'].iloc[-1])
        elif transf == 1:
            Y_hat_df['lstm_og'] = utilities.reconstruccion_log_diff(Y_hat_df['LSTM'], x_train['y'].iloc[-1])
        elif transf == 2:
            Y_hat_df['lstm_og'] = utilities.reconstruccion_pct(Y_hat_df['LSTM'], x_train['y'].iloc[-1])
        elif transf == 3:
            Y_hat_df['lstm_og'] = np.expm1(Y_hat_df['LSTM'])
        elif transf == 4:
            Y_hat_df['lstm_og'] = Y_hat_df['LSTM']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            Y_hat_df['lstm_og'] = utilities.reconstruccion_diff2(Y_hat_df['LSTM'], y_t_1, y_t_2)
    
        comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
        # MAE
        mae = mean_absolute_error(comparativa['y'], comparativa['lstm_og'])
        cv_mae.append(mae)
        # RMSE
        rmse = root_mean_squared_error(comparativa['y'], comparativa['lstm_og'])
        cv_rmse.append(rmse)
        # MAPE
        comparativa['mape'] = comparativa.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)
        total_mape = comparativa['mape'].mean()
        cv_mape.append(total_mape)
        # MSE
        mse = mean_squared_error(comparativa['y'], comparativa['lstm_og'])
        cv_mse.append(mse)

    if config['metric'] == 0:
        return -round(np.mean(cv_mape) , 3)
        
    elif config['metric'] == 1:
        return -round(np.mean(cv_mae) , 3)

    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)
        #tune.report({"error": mse})
    # WE are returning MAE. 
    # Objetive Funtion Construcction must Match 
    # The construcción of the Predict Function.
    # a slight change, such as the loss funcion ie. MAE, RMSE
    # Will cause totally different results.

# DeepAr
def obj_deep_ar_cv(data=None,
                years=None,
                months=None,
                h=None,
                input_size=None,
                layers=None,
                trajectories=None,
                learning_rate=None,
                max_steps=None,
                neurons=None,
                freq=None,
                metric=None,
                #Features
                feats=None,
                # Transformaciones
                signals=None,
                transf=None,
                batch_size=None
                ):
    # Por comodidad. Dado que se tenia establecido así para Ray Tune
    # Pero al realizar multiples pruebas, llegue a la conclusión de que 
    # Es puta mierda, tarda más de lo que debería para entrenar/ajustar modelos. 
    # Bayesian optimization has prove to be 70% quicker. 
    config = {'years':int(years),
            'months':int(months),
            'h':int(h),
            'input_size':int(input_size),
            'layers':int(layers),
            'trajectories':int(trajectories),
            'learning_rate':learning_rate,
            'max_steps':int(max_steps),
            'neurons':int(neurons),
            'freq':freq,
            'metric':metric,
            'feats':int(feats),
            'signals': int(signals),
            'transf':transf,
            'batch_size':batch_size
            }

    # Rango
    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    # Variables Temporales
    data = utilities.features_from_date(data, 'ds')
    # Transformación Senoidal d Variables Temporales
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    # Se junta el conjunto de validación para formar el gran conjunto, el cual será dividido en 4 splits con horizontes 
    # Previamente definidos por el usuario desde el modulo main.py 
    x_train_val = pd.concat([x_train, x_val])
    
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size'])
    # Horizonte
    horizonte = int(config['h'])

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []
    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]

        # Transformaciones
        if config['transf'] == 0:
            x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
            x_train = x_train.dropna()  #El primer valor será NaN

        elif config['transf'] == 1:
            x_train['y_log'] = np.log1p(x_train['y'])
            x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
            x_train = x_train.dropna()

        elif config['transf'] == 2:
            x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
            x_train = x_train.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3:
            x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

        elif config['transf'] == 4:
            x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

        elif config['transf'] == 5:
            x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
            x_train = x_train.dropna()
        
        # Estandarización
        transf = config['transf']
        mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
        sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
        x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu) / sigma
    
        # Escalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
        # Senoidales
        senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'], 
                                                                top_k=int(config['signals']),
                                                                extra_steps=config['h'])
        for c in senoidales.columns:
            x_train[c] = (senoidales[c]).values[:len(x_train)]

        

        # Variables Exogenas
        if config['feats']==0: # Ninguna
            features=None
        elif config['feats']==1: # Temporales
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        elif config['feats']==2: # Temporales + Senoidales
            # Features Senoidales added.
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
            features += list(senoidales.columns)[:-1]
    
        nf = NeuralForecast(
            models=[DeepAR(h=horizonte,
                        input_size=int(horizonte*config['input_size']),
                        lstm_n_layers=int(config['layers']),
                        trajectory_samples=int(config['trajectories']),
                        lstm_hidden_size=config['neurons'],
                        loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=False),
                        valid_loss=MQLoss(level=[80, 90]),
                        learning_rate=config['learning_rate'],
                        futr_exog_list=features,
                        max_steps=int(config['max_steps']),
                        val_check_steps=5,
                        windows_batch_size=config['batch_size'],
                        early_stop_patience_steps=15,
                        scaler_type='identity',
                        enable_progress_bar=False,
                        random_seed=119,
                        ),
            ],
            freq=frequency_map[config['freq']]
        )

        nf.fit(df=x_train, target_col='y_scaled', verbose=False, val_size=horizonte)
        
        # Variables Exogenas
        if config['feats']==0: # Ninguna
            Y_hat_df = nf.predict(verbose=0)
        
        elif config['feats']==1: # Temporales
            df_features = nf.make_future_dataframe()
            # Data Features
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        elif config['feats']==2: # Temporales + Senoidales
            df_features = nf.make_future_dataframe()
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            for c in senoidales.columns[:-1]:
                df_features[c] = (senoidales[c]).values[len(x_train):]
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)


        # Se Reconstruye la Inversa. 
        Y_hat_df['DeepAR'] = scaler.inverse_transform([Y_hat_df['DeepAR'].values])[0]
        Y_hat_df['DeepAR'] = (Y_hat_df['DeepAR']*sigma) + mu

        # Reconstruccion
        transf = config['transf']
        if transf == 0:
            Y_hat_df['deepAr_og'] = utilities.reconstruccion_diff(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
        elif transf == 1:
            Y_hat_df['lstm_og'] = utilities.reconstruccion_log_diff(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
        elif transf == 2:
            Y_hat_df['deepAr_og'] = utilities.reconstruccion_pct(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
        elif transf == 3:
            Y_hat_df['deepAr_og'] = np.expm1(Y_hat_df['DeepAR'])
        elif transf == 4:
            Y_hat_df['deepAr_og'] = Y_hat_df['DeepAR']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            Y_hat_df['deepAr_og'] = utilities.reconstruccion_diff2(Y_hat_df['DeepAR'], y_t_1, y_t_2)
    
        comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
        # MAE
        mae = mean_absolute_error(comparativa['y'], comparativa['deepAr_og'])
        cv_mae.append(mae)
        # RMSE
        rmse = root_mean_squared_error(comparativa['y'], comparativa['deepAr_og'])
        cv_rmse.append(rmse)
        # MAPE
        comparativa['mape'] = comparativa.apply(accuracy.mape, args=('deepAr_og', 'y'), axis=1)
        total_mape = comparativa['mape'].mean()
        cv_mape.append(total_mape)
        # MSE
        mse = mean_squared_error(comparativa['y'], comparativa['deepAr_og'])
        cv_mse.append(mse)

    if config['metric'] == 0:
        return -round(np.mean(cv_mape) , 3)
        
    elif config['metric'] == 1:
        return -round(np.mean(cv_mae) , 3)

    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)

# Transformer
def obj_transformer_cv(data=None,
                    years=None,
                    months=None,
                    h=None,
                    input_size=None,
                    neurons=None,
                    conv_size=None,
                    n_heads=None,
                    max_steps=None,
                    freq=None,
                    metric=None,
                    learning_rate=None,
                    # signals
                    signals=None,
                    #Features,
                    feats=None,
                    # Transformaciones
                    transf=None,
                    batch_size=None
                    ):

    config = {'years':int(years),
            'months':int(months),
            'h':int(h),
            'input_size':int(input_size),
            'conv_size':int(conv_size),
            'n_heads':int(n_heads),
            'max_steps':int(max_steps),
            'neurons':int(neurons),
            'freq':freq,
            'metric':metric,
            'feats':int(feats),
            'learning_rate':learning_rate,
            'signals': int(signals),
            'transf':transf,
            'batch_size':batch_size
            }

    # Rango
    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    # Variables Temporales
    data = utilities.features_from_date(data, 'ds')
    # Transformación Senoidal d Variables Temporales
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    # Se junta el conjunto de validación para formar el gran conjunto, el cual será dividido en 4 splits con horizontes 
    # Previamente definidos por el usuario desde el modulo main.py 
    x_train_val = pd.concat([x_train, x_val])
    
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    config['conv_size'] = 2 ** int(config['conv_size'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size'])

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []

    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]

        # Transformaciones
        if config['transf'] == 0:
            x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
            x_train = x_train.dropna()  #El primer valor será NaN

        elif config['transf'] == 1:
            x_train['y_log'] = np.log1p(x_train['y'])
            x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
            x_train = x_train.dropna()

        elif config['transf'] == 2:
            x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
            x_train = x_train.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3:
            x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

        elif config['transf'] == 4:
            x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

        elif config['transf'] == 5:
            x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
            x_train = x_train.dropna()
        
        # Estandarización
        transf = config['transf']
        mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
        sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
        x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu) / sigma
    
        # Escalar
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
        # Senoidales
        senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'], 
                                                                top_k=int(config['signals']),
                                                                extra_steps=config['h'])
        for c in senoidales.columns:
            x_train[c] = (senoidales[c]).values[:len(x_train)]

        # Horizonte
        horizonte = config['h']

        # Variables Exogenas
        if config['feats']==0: # Ninguna
            features=None
        elif config['feats']==1: # Temporales
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        elif config['feats']==2: # Temporales + Senoidales
            # Features Senoidales added.
            features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos']]
            features += list(senoidales.columns)[:-1]

        nf = NeuralForecast(
            models=[VanillaTransformer(h=horizonte,
                                    input_size=horizonte*int(config['input_size']),
                                    hidden_size=config['neurons'],
                                    conv_hidden_size=config['conv_size'],
                                    n_head=int(config['n_heads']),
                                    loss=metric_map[config['metric']],
                                    valid_loss=metric_map[config['metric']],
                                    futr_exog_list=features,
                                    scaler_type='standard',
                                    learning_rate=config['learning_rate'],
                                    max_steps=int(config['max_steps']),
                                    val_check_steps=5,
                                    windows_batch_size=config['batch_size'],
                                    early_stop_patience_steps=25,
                                    enable_progress_bar=False,
                                    start_padding_enabled=False,
                                    random_seed=119),
            ],
            freq=frequency_map[config['freq']]
        )
        try:
            nf.fit(df=x_train, 
                   target_col='y_scaled', 
                   verbose=False, 
                   val_size=horizonte)
        except Exception as e:
            return -10
            print(e)

        # Variables Exogenas
        if config['feats']==0: # Ninguna
            Y_hat_df = nf.predict(verbose=0)
        
        elif config['feats']==1: # Temporales
            df_features = nf.make_future_dataframe()
            # Data Features
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        elif config['feats']==2: # Temporales + Senoidales
            df_features = nf.make_future_dataframe()
            df_features = utilities.features_from_date(df_features, 'ds')
            for col, max_val in cyclic_cols.items():
                df_features = utilities.add_cyclic_features(df_features, col, max_val)
            for c in senoidales.columns[:-1]:
                df_features[c] = (senoidales[c]).values[len(x_train):]
            Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

        # Se Reconstruye la Inversa. 
        Y_hat_df['VanillaTransformer'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
        Y_hat_df['VanillaTransformer'] = (Y_hat_df['VanillaTransformer']*sigma) + mu

        # Reconstruccion
        transf = config['transf']
        if transf == 0:
            Y_hat_df['Transformer_og'] = utilities.reconstruccion_diff(Y_hat_df['VanillaTransformer'], x_train['y'].iloc[-1])
        elif transf == 1:
            Y_hat_df['Transformer_og'] = utilities.reconstruccion_log_diff(Y_hat_df['VanillaTransformer'], x_train['y'].iloc[-1])
        elif transf == 2:
            Y_hat_df['Transformer_og'] = utilities.reconstruccion_pct(Y_hat_df['VanillaTransformer'], x_train['y'].iloc[-1])
        elif transf == 3:
            Y_hat_df['Transformer_og'] = np.expm1(Y_hat_df['VanillaTransformer'])
        elif transf == 4:
            Y_hat_df['Transformer_og'] = Y_hat_df['VanillaTransformer']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            Y_hat_df['Transformer_og'] = utilities.reconstruccion_diff2(Y_hat_df['VanillaTransformer'], 
                                                                        y_t_1, y_t_2)
    
        comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
        # MAE
        mae = mean_absolute_error(comparativa['y'], comparativa['Transformer_og'])
        cv_mae.append(mae)
        # RMSE
        rmse = root_mean_squared_error(comparativa['y'], comparativa['Transformer_og'])
        cv_rmse.append(rmse)
        # MAPE
        comparativa['mape'] = comparativa.apply(accuracy.mape, args=('Transformer_og', 'y'), axis=1)
        total_mape = comparativa['mape'].mean()
        cv_mape.append(total_mape)
        # MSE
        mse = mean_squared_error(comparativa['y'], comparativa['Transformer_og'])
        cv_mse.append(mse)

    if config['metric'] == 0:
        return -round(np.mean(cv_mape) , 3)
        
    elif config['metric'] == 1:
        return -round(np.mean(cv_mae) , 3)

    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)

# D3VAE
def obj_dvae_cv(data=None,
            years=None,
            months=None,
            h=None,
            feats=None,
            signals=None,
            #use_fourier=None,
            input_size=None,
            max_steps=None,
            neurons=None,
            layers=None,
            dropout=None,
            beta_kl=None,
            teacher_forcing=None,
            batch_size=None,
            transf=None,
            dimension=None, 
            metric=None, # The one we want to maximize
            freq=None # Defines de freakuency
            ):
    
    config = {'years':years,
        'months':months,
        'h':int(h),
        'input_size':int(input_size),
        'neurons':neurons,
        'layers':layers,
        'max_steps':max_steps,
        'freq':freq,
        'metric':metric,
        'feats':int(feats),
        'signals': int(signals),
        'transf':transf,
        'batch_size': int(batch_size),
        'feats':int(feats),
        'dropout':dropout,
        'beta_kl':beta_kl,
        'teacher_forcing':teacher_forcing,
        'dimension':dimension,
    }
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])

    # conversión de Batch Size.
    # 2^1 = 2, ..., 2^5= 32, 2^6= 64, 2^7= 128, 2^8= 256
    config['batch_size'] = 2 ** int(config['batch_size'])

    # Conversion de Dimensión.
    # Un buen rango es entre 8, 16 y 32. More than that, seems like overfitting to me
    # (And according to the tests I have ran.) So it would be (2, 6)
    config['dimension'] = 2 ** int(config['dimension'])

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data,
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    # Se junta el conjunto de validación para formar el gran conjunto, el cual será dividido en 4 splits con horizontes 
    # Previamente definidos por el usuario desde el modulo main.py 
    x_train_val = pd.concat([x_train, x_val])
    
    
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())

    if config['feats'] == 0:
        use_time_features=False
        use_fft_features=False

    elif config['feats'] == 1:
        use_time_features=True
        use_fft_features=False

    elif config['feats'] == 2:
        use_time_features=True
        use_fft_features=True

    cv_i = 0
    cv_mae = []
    cv_rmse = []
    cv_mape = []
    cv_mse = []

    for dt_cv in simulation_dates_cv[:-1]:
        cv_i = cv_i + 1 
        x_train = x_train_val[x_train_val['ds']<dt_cv]
        x_val =  x_train_val[ (x_train_val['ds']>=dt_cv) & (x_train_val['ds']<simulation_dates_cv[cv_i])]
        
        # Transformaciones
        if config['transf'] == 0:
            x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
            x_train = x_train.dropna()  #El primer valor será NaN

        elif config['transf'] == 1:
            x_train['y_log'] = np.log1p(x_train['y'])
            x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
            x_train = x_train.dropna()

        elif config['transf'] == 2:
            x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
            x_train = x_train.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3:
            x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

        elif config['transf'] == 4:
            x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

        elif config['transf'] == 5:
            x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
            x_train = x_train.dropna()

        # Estandarización
        transf = config['transf']
        mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
        sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
        x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu) / sigma

        str_datee = x_train.ds.max()
        str_datee = str(str_datee)[: 10]
        x_train[['unique_id', 'ds', 'y_estandarizada']].rename(
            columns={'y_estandarizada':'y'}
        ).to_csv('temporal_file_d3vae.csv')

        print(x_train['unique_id'].unique()[0])
        id_train = str(x_train['unique_id'].unique()[0])
        #pr

        #config['h'] =  config['h']
        
        config_d3vae = {
            'data': {
                'csv_path': 'temporal_file_d3vae.csv',
                'context_len': config['h']*config['input_size'],
                'horizon': config['h'],
                'stride': 1,
                'batch_size': int(config['batch_size']),
                'num_workers': 0,
                'val_split': 0.2,
                'normalize': 'zscore',
                'models_to_train': ["d3vae"],
                'minmax_range': [0.0, 1.0],
                'time_features': ["dow", "month", "weekofyear", "dayofyear"],   # produce sin/cos por cada una "dow", 
                'use_time_features': use_time_features,
                'use_fourier': False,
                'use_fft_features': use_fft_features,
                'fft_top_k': config['signals'],
                'fft_scale_range': [0.0, 1.0], # escala de las ondas
                'fft_include_signal': False,# agrega la señal reconstruida como extra, en general, es mejor no incluirla por que 
                                            # Leakea información pasada, y hace que los modelos aprendan a seguir la señal, la cual
                                            # Será erronea a futuro. Es mejor aprendan sin la señal. 
                'fft_window': "hann",       # opcional: "hann" o null, hann por default.
                'freq': frequency_map[config['freq']], # El sistema puede inferirla, pero es mejor definirla.
                'split_mode': 'time',   # <- usa holdout temporal por serie
                'cutoff': str_datee, #'2023-09-18',
                'from_date': '2000-01-01',
                'id': 'Aguascalientes_TMAX' #id_train
            },

            #'d3vae':{
            #    'T': 50,
            #    'schedule': "cosine",         # o "cosine" / "linear"
            #    'beta_x': [1e-4, 2e-2],
            #    'beta_y': [1e-4, 2e-2],
            #    'time_emb_dim': 32+12,
            #    'dsm_weight': 0.2,
            #    'tc_weight': 0.1,
            #    'jump_gamma': 0.0,
            #    'jump_t': 0,
            #},

            'model': {
                'input_size': 1,
                'latent_dim': int(config['dimension']),  # 32
                'enc_hidden': int(config['neurons']),    # 128
                'enc_layers': int(config['layers']),     # 3
                'dec_hidden': int(config['neurons']),    # 128
                'dec_layers': int(config['layers']),     # 3
                'dropout': round(float(config['dropout']), 3),       # .1
                'beta_kl': round(float(config['beta_kl']), 3),#config['beta_kl'],       # .5
                'teacher_forcing': round(float(config['teacher_forcing'])),#0.2,
                'predict_sigma': False
            },

            'train': {
                #'beta_kl': 1,
                #'kl_warmup_epochs': 0, # estaba en 7
                #'teacher_forcing_start': 0.7,
                #'teacher_forcing_end': 0.1,
                'early_stop_patience': 20,
                'epochs': int(config['max_steps']),
                'lr': 1e-3,
                'weight_decay': 1e-5,
                'grad_clip': 1.0,
                'seed': 119,
                'device': 'auto',
                'save_dir': 'runs/dvae_v2'
            }
        }

        with open('d3vae_config.yaml', 'w') as f:
            yaml.safe_dump(config_d3vae, f)
        # Se entrena el Modelo
        train_main('d3vae_config.yaml')
        # Se carga la configuración. # Pero esto es más para temas de visualización.     
        cfg = yaml.safe_load(open('d3vae_config.yaml'))
        #ctx = cfg['data']['context_len']; H = cfg['data']['horizon']

        predict_main('d3vae_config.yaml', 'runs/dvae_v2/d3vae/best.ckpt ', f'temporal_file_d3vae.csv', 100, 'auto')
        preds = pd.read_csv(f'temporal_file_d3vae.csv')
        preds['ds'] = pd.to_datetime(preds['ds'])
        preds.rename(columns={'mean':'d3vae'}, inplace=True)

        # Se Reconstruye la Inversa. 
        #Y_hat_df['VanillaTransformer'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
        preds['d3vae'] = (preds['d3vae']*sigma) + mu

        # Reconstruccion
        transf = config['transf']
        if transf == 0:
            preds['d3vae'] = utilities.reconstruccion_diff(preds['d3vae'], x_train['y'].iloc[-1])
        elif transf == 1:
            preds['d3vae'] = utilities.reconstruccion_log_diff(preds['d3vae'], x_train['y'].iloc[-1])
        elif transf == 2:
            preds['d3vae'] = utilities.reconstruccion_pct(preds['d3vae'], x_train['y'].iloc[-1])
        elif transf == 3:
            preds['d3vae'] = np.expm1(preds['d3vae'])
        elif transf == 4:
            preds['d3vae'] = preds['d3vae']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            preds['d3vae'] = utilities.reconstruccion_diff2(preds['d3vae'], 
                                                                        y_t_1, y_t_2)

        comparativa = preds.merge(x_val, on=['unique_id', 'ds'], how='inner')
        y_pred = comparativa['d3vae'].to_numpy()
        if utilities.is_linear_by_fit(y_pred, tol=.1):
            pen_l = 5  # castigar al modelo
        else:
            pen_l = 0
        
        '''plt.figure(figsize=(12, 8))
        plt.plot(comparativa['ds'], 
                comparativa['y'], #marker='o', 
                label='Inflación')
        plt.plot(comparativa['ds'], 
                comparativa['d3vae'], #marker='o', 
                label='d3vae')
        plt.title(f"Inflación Anual ~ D3VAE- Validación Cruzada" )
        plt.xlabel("Fecha")
        plt.ylabel("Tasa Anual")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()'''
        # MAE
        mae = mean_absolute_error(comparativa['y'], comparativa['d3vae'])
        cv_mae.append(mae+pen_l)
        # RMSE
        rmse = root_mean_squared_error(comparativa['y'], comparativa['d3vae'])
        cv_rmse.append(rmse)
        # MAPE
        comparativa['mape'] = comparativa.apply(accuracy.mape, args=('d3vae', 'y'), axis=1)
        total_mape = comparativa['mape'].mean()
        cv_mape.append(total_mape)
        # MSE
        mse = mean_squared_error(comparativa['y'], comparativa['d3vae'])
        cv_mse.append(mse)

    if config['metric'] == 0:
        return -round(np.mean(cv_mape) , 3)
        
    elif config['metric'] == 1:
        return -round(np.mean(cv_mae) , 3)

    elif config['metric'] == 2:
        return -round(np.mean(cv_rmse) , 3)

    elif config['metric'] == 3:
        return -round(np.mean(cv_mse) , 3)

