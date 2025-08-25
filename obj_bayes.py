# obj bayes opt


# Obj Functions
# tune Example 
from ray import tune
from functools import partial
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

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, VanillaTransformer, LSTM, RNN, NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE, MSE
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
    1: MAE(),
    0: MAPE(),
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
import process_data
inpc_path_Q = 'ca56_2018a.csv'
inpc_Q = process_data.limpiar_csv_inegi(inpc_path_Q)
weekly_inpc = process_data.inpc_data_weekly(datos=inpc_Q)

def naive_improvement_penalty(y_true, y_last, y_pred ,s_recent=None, gamma=0.5):
    """Penaliza si NO mejoras al naive."""
    H = len(y_true)
    h = np.arange(1, H+1)
    if s_recent is None: s_recent = 0.0
    y_naive = y_last + s_recent*h
    imp = mean_absolute_error(y_true, y_naive) - mean_absolute_error(y_true, y_pred)  # >0: mejoras
    return float(gamma * max(0.0, -imp))  # penaliza solo si no mejoras


def obj_fft(data=None,
            years=None,
            months=None,
            metric=None,
            transf=None,
            signals=None,
            h=None,
            freq=None
            ):
    
    config = {
            # Temporales 
            'years':years,
            'months':months,
            'metric':metric,
            'h':int(h),
            'signals':int(signals),
            'transf':transf,
            'freq':freq}
    
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
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

    # Transformas la variable, la escalas, construyes la señal usando n senoidales top, 
    #  y luego la señal la re-escalas, y la reconstruyes para compararla contra 
    # la variable verdadera. 

    # Escalas el conjunto de entrenamiento antes de generar las senoidales y reconstruir la señal. 
    scaler = MinMaxScaler(feature_range=(0, 1))
    #x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    x_train['y_scaled'] = scaler.fit_transform(x_train[f'y_{inverse_map_tr[transf]}'].values.reshape(-1, 1))

    senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_scaled'],
                                                            top_k=config['signals'],
                                                            extra_steps=config['h']+ len(x_val))
    
    for c in senoidales.columns:
        x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))]
    x_val['signal'] = scaler.inverse_transform([x_val['signal'].values])[0]
    
    transf = config['transf']
    if transf == 0:
        x_val['yhat_og'] = utilities.reconstruccion_diff(x_val['signal'], x_train['y'].iloc[-1])
    elif transf == 1:
        x_val['yhat_og'] = utilities.reconstruccion_log_diff(x_val['signal'], x_train['y'].iloc[-1])
    elif transf == 2:
        x_val['yhat_og'] = utilities.reconstruccion_pct(x_val['signal'], x_train['y'].iloc[-1])
    elif transf == 3:
        x_val['yhat_og'] = np.expm1(x_val['signal'])
    elif transf == 4:
        x_val['yhat_og'] = x_val['signal']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        x_val['yhat_og'] = utilities.reconstruccion_diff2(x_val['signal'], y_t_1, y_t_2)
    
    #mape_total = x_val['mape'].mean()
    try:
        # MAE
        mae = mean_absolute_error(x_val['y'], x_val['yhat_og'])
        # RMSE
        rmse = root_mean_squared_error(x_val['y'], x_val['yhat_og'])
        # MSE
        mse = mean_squared_error(x_val['y'], x_val['yhat_og'])

        # With Bayesian optimization we return the error, but since we maximizing
        # we have to send a negative version of it. 
        if config['metric'] == 0:
            return -mape_total
            #tune.report({"error": mape_total})

        elif config['metric'] == 1:
            return -mae
            #tune.report({"error": mae})

        elif config['metric'] == 2:
            return -rmse
            #tune.report({"error": rmse})

        elif config['metric'] == 3:
            return -mse

    except Exception as e:
        return -10000


def obj_holt_winters(data=None,
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

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
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

    # Holt-Winters Parameters
    if config['trend_type'] >= .5:
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
    #Val = Val.set_index('ds')

    preds_gen = H.merge(weekly_inpc, on=['unique_id', 'ds'], how='left')
    preds_gen.dropna(inplace=True)
    try:
        print(mean_absolute_error(preds_gen['y'], preds_gen['HoltW_og']))
    except Exception as e:
        print(e)
        print(preds_gen.head(2))
        print(preds_gen.tail(2))
        
    x_val = x_val.merge(H, on=['ds', 'unique_id'], how='inner')

    #x_val['yhat_og'].fillna(0, inplace=True)
    #x_val['yhat_og'] = x_val['yhat'].clip(lower=0)
    #print(x_val)
    #Val['yhat'] = Val['yhat'].astype(int)

    # MAPE
    x_val['mape'] = x_val.apply(accuracy.mape, args=('HoltW_og','y'), axis=1)
    mape_total = x_val['mape'].mean()
    try:
        # MAE
        mae = mean_absolute_error(x_val['y'], x_val['HoltW_og'])
        # RMSE
        rmse = root_mean_squared_error(x_val['y'], x_val['HoltW_og'])
        # MSE
        mse = mean_squared_error(x_val['y'], x_val['HoltW_og'])

        # With Bayesian optimization we return the error, but since we maximizing
        # we have to send a negative version of it. 
        if config['metric'] == 0:
            return -mape_total
            #tune.report({"error": mape_total})

        elif config['metric'] == 1:
            return -mae
            #tune.report({"error": mae})

        elif config['metric'] == 2:
            return -rmse
            #tune.report({"error": rmse})

        elif config['metric'] == 3:
            return -mse

    except Exception as e:
        return -10000
        #tune.report({"error": mse})

## Seasonal Naive
# BASELINE

# Machine Learning

# XGBoost
# Hacer Distintos Experimentos para evaluar
# Si combiene meter más variables exogenas
# Que en este caso serán variables senoidales
# O si es mejor dejarlo así. Osea, un A/B Test.
def obj_xgb(years=None,
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
            transf=None):
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
        for col, max_val in cyclic_cols.items():
            data = utilities.add_cyclic_features(data, col, max_val)
        # Transformación Senoidal d Variables Temporales
        #feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]

        # Transformaciones
        if config['transf'] == 0: # Diff 1
            data[f'y_{inverse_map_tr[0]}'] = data['y'].diff()
            data = data.dropna()  #El primer valor será NaN

        elif config['transf'] == 1: # Logp1 Diff 
            data['y_log'] = np.log1p(data['y'])
            data[f'y_{inverse_map_tr[1]}'] = data['y_log'].diff()
            data = data.dropna()

        elif config['transf'] == 2:
            data[f'y_{inverse_map_tr[2]}'] = data['y'].pct_change() * 100
            data = data.dropna()  #El primer valor será NaN
            
        elif config['transf'] == 3: # Logp1
            data[f'y_{inverse_map_tr[3]}'] = np.log1p(data['y'])

        elif config['transf'] == 4: # none
            data[f'y_{inverse_map_tr[4]}'] = data['y']

        elif config['transf'] == 5: # Double Trouble
            data[f'y_{inverse_map_tr[5]}'] = data['y'].diff().diff()
            data = data.dropna()

        # División de Conjuntos.
        x_train, x_val = utilities.split_data_val(data=data, 
                                                train_years=int(config['years']), 
                                                months_val=int(config['months']), 
                                                date='ds')
        
        #print(x_train)
        
        # XGBoost only has 2 options, unless lags are added.
        if config['feats']==0 or config['feats']==1:
            feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        elif config['feats']==2:# Features Senoidales added.
            # Senoidales
            senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'],
                                                                    top_k=config['signals'],
                                                                    extra_steps=config['h']+ len(x_val))
            # Entrenamiento
            for c in senoidales.columns:
                x_train[c] = (senoidales[c]).values[:len(x_train)]
            # Validación
            for c in senoidales.columns:
                x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))]
            feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
            feats += list(senoidales.columns)[:-1]

        # Calcular la correlación entre las series
        #corr_matrix = x_train.drop(columns=["ds", 'unique_id']).corr()

        '''# Plot del heatmap
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Matriz de Correlación entre Series Temporales")
        plt.show()'''
        
        #print(feats)
        # Matrices de XGBoost
        dtrain = xgb.DMatrix(x_train[feats], label=x_train[f'y_{inverse_map_tr[transf]}'], feature_names=feats)
        dval = xgb.DMatrix(x_val[feats], label=x_val[f'y_{inverse_map_tr[transf]}'], feature_names=feats)

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
            'gamma':.1,
            'max_bin':512,
            'eval_metric':"mae"#'mape'
        }
        # Train XGBoost model on training set
        xgb_model_train = xgb.train(
            param,
            dtrain,
            num_boost_round=config['num_boost_round'],
            early_stopping_rounds=50,
            verbose_eval=False,
            evals=[(dval, 'val')])

        # Evaluate model on test set and return score
        x_val['yhat'] = xgb_model_train.predict(dval)

        transf = config['transf']
        if transf == 0:
            x_val['yhat_og'] = utilities.reconstruccion_diff(x_val['yhat'], x_train['y'].iloc[-1])
        elif transf == 1:
            x_val['yhat_og'] = utilities.reconstruccion_log_diff(x_val['yhat'], x_train['y'].iloc[-1])
        elif transf == 2:
            x_val['yhat_og'] = utilities.reconstruccion_pct(x_val['yhat'], x_train['y'].iloc[-1])
        elif transf == 3:
            x_val['yhat_og'] = np.expm1(x_val['yhat'])
        elif transf == 4:
            x_val['yhat_og'] = x_val['yhat']
        elif transf == 5:
            y_t_1 = x_train['y'].iloc[-1]
            y_t_2 = x_train['y'].iloc[-2]
            x_val['yhat_og'] = utilities.reconstruccion_diff2(x_val['yhat'], y_t_1, y_t_2)


        x_val['mape'] = x_val.apply(accuracy.mape, args=('yhat_og','y'), axis=1)
        mape_total = x_val['mape'].mean()
        
        # MAE
        mae = mean_absolute_error(x_val['y'], x_val['yhat_og'])
        # RMSE
        rmse = root_mean_squared_error(x_val['y'], x_val['yhat_og'])
        # MSE
        mse = mean_squared_error(x_val['y'], x_val['yhat_og'])

        # With Bayesian optimization we return the error, but since we maximizing
        # we have to send a negative version of it. 
        if config['metric'] == 0:
            return -mape_total
            #tune.report({"error": mape_total})

        elif config['metric'] == 1:
            return -mae
            #tune.report({"error": mae})

        elif config['metric'] == 2:
            return -rmse
            #tune.report({"error": rmse})

        elif config['metric'] == 3:
            return -mse
            #tune.report({"error": mse})
    except Exception as e:
        print(e)
        return -1000

# Neural Networks

# Se usa Nixtla por que está tan bien implementado y optimizado, que realmente se pueden obtener 
# Buenos resultados a partir de la construcción de modelos.
# Además de ahorrar tiempo, y posibles sesgos al construir las ventanas de tiempo.
# al final del día, fue construido para acelerar procesos y no usarlo sería ilogico. 
# RNN
def obj_rnn(data=None,
            years=None,
            months=None,
            h=None,
            input_size=None,
            neurons=None,
            layers=None,
            max_steps=None,
            freq=None,
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
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])

    # Horizonte
    horizonte = config['h'] + len(x_val)

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
                    input_size=horizonte*config['input_size'],
                    # Metricas de Evaluación
                    loss=metric_map[config['metric']],
                    valid_loss=metric_map[config['metric']],
                    # Escalamiento de Datos & Posibles variables exogenas
                    scaler_type='standard',
                    encoder_n_layers=int(config['layers']),
                    encoder_hidden_size=config['neurons'],
                    decoder_hidden_size=config['neurons'],
                    decoder_layers=int(config['layers']),
                    max_steps=int(config['max_steps']),
                    futr_exog_list=features,
                    #stat_exog_list=['airline1'],
                    enable_progress_bar=False,
                    #start_padding_enabled=True
                    random_seed=119
                    )
        ],
        freq=frequency_map[config['freq']]
    )

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)

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
    preds_gen = Y_hat_df.merge(weekly_inpc, on=['unique_id', 'ds'], how='left')
    preds_gen.dropna(inplace=True)
    print(mean_absolute_error(preds_gen['y'], preds_gen['rnn_og']))

    #print(comparativa[['ds', 'y', 'rnn_og', 'RNN']])
    #input()
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['rnn_og'])
    if mae<.3:
        mae = .9

    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['rnn_og'])
    # MAPE
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('rnn_og', 'y'), axis=1)
    # MSE
    mse = mean_squared_error(comparativa['y'], comparativa['rnn_og'])

    total_mape = comparativa['mape'].mean()

    # With Bayesian optimization we return the error, but since we maximizing
    # we have to send a negative version of it. 
    if config['metric'] == 0:
        return -total_mape
        #tune.report({"error": mape_total})

    elif config['metric'] == 1:
        return -mae
        #tune.report({"error": mae})

    elif config['metric'] == 2:
        return -rmse
        #tune.report({"error": rmse})

    elif config['metric'] == 3:
        return -mse
        #tune.report({"error": mse})

# LSTM
def obj_lstm(data=None,
            years=None,
            months=None,
            h=None,
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
            transf=None
            ):

    config={'years':years,
            'months':months,
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
    
    if config['transf'] == 0: # Simple Diff
        x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
        x_train = x_train.dropna()  #El primer valor será NaN

    elif config['transf'] == 1: # Logp1 Diff
        x_train['y_log'] = np.log1p(x_train['y'])
        x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
        x_train = x_train.dropna()

    elif config['transf'] == 2: # PCT
        x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
        x_train = x_train.dropna()  #El primer valor será NaN
        
    elif config['transf'] == 3: # Logp1
        x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

    elif config['transf'] == 4: # None
        x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

    elif config['transf'] == 5: # Double Diff
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
                                                            extra_steps=config['h']+ len(x_val))
    
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Horizonte
    horizonte = config['h'] + len(x_val)

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
                    # Metricas
                    loss=metric_map[config['metric']],
                    valid_loss=metric_map[config['metric']],
                    # Learning Rate
                    #learning_rate=config['learning_rate'],#0.001,
                    #stat_exog_list=['tipo'],
                    futr_exog_list=features,
                    max_steps=int(config['max_steps']),
                    #early_stop_patience_steps=-1,
                    scaler_type='standard',
                    enable_progress_bar=False,
                    random_seed=119
                    #start_padding_enabled=True
                    ),
        ],
        freq=frequency_map[config['freq']]
    )
    #print('MAMO?')

    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    
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
    
    # Se recupera la escala.
    Y_hat_df['LSTM'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    
    # Y_hat_df['LSTM']  = (Y_hat_df['LSTM']*sigma)+mu # Estandarización

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
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)

    preds_val =  Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='left')
    preds_gen = Y_hat_df.merge(weekly_inpc, on=['unique_id', 'ds'], how='left')
    preds_gen.dropna(inplace=True)
    print(mean_absolute_error(preds_gen['y'], preds_gen['lstm_og']))
    '''fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    #recent = performance[performance['ds']>'2020-01-01']
    plt.plot(preds_gen['ds'], preds_gen['y'], marker='o', label='inflacion')
    plt.plot(preds_gen['ds'], preds_gen['lstm_og'], marker='o', label='lstm_og')
    # --- línea vertical del corte ---
    plt.axvline(pd.to_datetime(x_val['ds'].max()), color='k', linestyle='--', linewidth=1.5, label='Corte validación')

    plt.title(f"Inflación mensual - Feature - fecha ~ Señales" )
    plt.xlabel("Fecha")
    plt.ylabel("Porcentaje")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()'''

    # MAPE
    total_mape = comparativa['mape'].mean()
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['lstm_og'])
    # ---- dentro de obj_lstm, tras construir 'comparativa' ----

    y_true = comparativa['y'].to_numpy()
    y_pred = comparativa['lstm_og'].to_numpy()

    mae   = mean_absolute_error(y_true, y_pred)
    pen_s = utilities.slope_match_penalty(y_true, y_pred, delta=None, schedule="linear",
                                asym_up=1.0, asym_down=1.6)
    pen_t = utilities.turn_penalty(y_true, y_pred)
    pen_c = utilities.curvature_match_penalty(y_true, y_pred) # pequeño
    pen_l = utilities.linearity_penalty(y_true, y_pred)  # dispara si ŷ ≈ recta
    # (opcional) guardrail contra perder vs naive (usa x_train disponible en tu función)
    y_last   = float(x_train['y'].iloc[-1])
    k = 6; t = np.arange(k)
    s_recent = np.polyfit(t, x_train['y'].tail(k).to_numpy(), 1)[0]
    pen_naive = naive_improvement_penalty(y_true, y_last, y_pred, s_recent, gamma=0.4)

    #print(pen_s)
    #print(pen_t)
    #print(pen_l)

    #score = -(mae + 0.8*pen_s + 0.5*pen_t + 0.08*pen_c + 10*pen_l + pen_naive)
    score = -(mae + 10*pen_l + pen_naive)
    return score

    y_true = comparativa['y'].to_numpy()
    y_pred = comparativa['lstm_og'].to_numpy()

    mae  = mean_absolute_error(y_true, y_pred)
    pen_slope = utilities.slope_match_penalty(y_true, y_pred,
                                    delta=None, schedule="linear",
                                    # TU CASO: LSTM baja demasiado (d_p < d_y) ⇒ penaliza más el 'dn'
                                    asym_up=1.0, asym_down=1.4)
    pen_turn  = utilities.turn_penalty(y_true, y_pred, tau=None, schedule="linear")
    pen_curv  = utilities.curvature_match_penalty(y_true, y_pred, delta=None)  # opcional

    # pesos recomendados para TU patrón:
    lam_slope = 4   # súbelo si aún persiste la caída lineal
    lam_turn  = 1.4   # fuerza a respetar el giro (signo)
    lam_curv  = 0.08  # bajito; sólo refuerza la U real

    score = -(mae + lam_slope*pen_slope + lam_turn*pen_turn + lam_curv*pen_curv)
    if score>-.2:
        score = -.9
    return score
    
    if mae<.3:
        mae = .9
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['lstm_og'])
    # MSE
    mse = mean_squared_error(comparativa['y'], comparativa['lstm_og'])

    # With Bayesian optimization we return the error, but since we maximizing
    # we have to send a negative version of it. 
    if config['metric'] == 0:
        return -total_mape
        #tune.report({"error": mape_total})

    elif config['metric'] == 1:
        return -mae
        #tune.report({"error": mae})

    elif config['metric'] == 2:
        return -rmse
        #tune.report({"error": rmse})

    elif config['metric'] == 3:
        return -mse
        #tune.report({"error": mse})
    # WE are returning MAE. 
    # Objetive Funtion Construcction must Match 
    # The construcción of the Predict Function.
    # a slight change, such as the loss funcion ie. MAE, RMSE
    # Will cause totally different results.

# DeepAr
def obj_deep_ar(data=None,
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
                transf=None
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
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)
    # Transformación Senoidal d Variables Temporales
    feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    # Transformaciones
    if config['transf'] == 0: # Simple Diff
        x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
        x_train = x_train.dropna()  #El primer valor será NaN

    elif config['transf'] == 1: # Logp1 Diff
        x_train['y_log'] = np.log1p(x_train['y'])
        x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
        x_train = x_train.dropna()

    elif config['transf'] == 2: # PCT
        x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
        x_train = x_train.dropna()  #El primer valor será NaN
        
    elif config['transf'] == 3: # Logp1
        x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

    elif config['transf'] == 4: # None
        x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

    elif config['transf'] == 5: # Double Diff
        x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
        x_train = x_train.dropna()

    # Estandarización
    transf = config['transf']
    mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
    sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
    x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu)/sigma

    # Escalar
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Senoidales
    senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train['y'],#x_train[f'y_{inverse_map_tr[transf]}'],
                                                            top_k=config['signals'], 
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Horizonte
    horizonte = config['h'] + len(x_val)

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
                    input_size=horizonte*int(config['input_size']),
                    lstm_n_layers=int(config['layers']),
                    trajectory_samples=int(config['trajectories']),
                    lstm_hidden_size=config['neurons'],
                    loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=False),
                    valid_loss=MQLoss(level=[80, 90]),
                    learning_rate=config['learning_rate'],#0.005,
                    #stat_exog_list=['airline1'],
                    # Features
                    futr_exog_list=features,
                    max_steps=int(config['max_steps']),
                    val_check_steps=50,
                    early_stop_patience_steps=-1,
                    scaler_type='identity',
                    enable_progress_bar=False,
                    random_seed=119,
                    #start_padding_enabled=True
                    ),
        ],
        freq=frequency_map[config['freq']]
    )
    try:
        nf.fit(df=x_train, target_col='y_scaled', verbose=False)
        
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

    except Exception as e:
        print(e)
        return -1000
    
    torch.cuda.empty_cache()

    Y_hat_df['DeepAR'] = scaler.inverse_transform([Y_hat_df['DeepAR'].values])[0]
    Y_hat_df['DeepAR']  = (Y_hat_df['DeepAR']*sigma)+mu

    # Reconstruccion
    transf = config['transf']
    if transf == 0:
        Y_hat_df['DeepAr_og'] = utilities.reconstruccion_diff(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
    elif transf == 1:
        Y_hat_df['DeepAr_og'] = utilities.reconstruccion_log_diff(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
    elif transf == 2:
        Y_hat_df['DeepAr_og'] = utilities.reconstruccion_pct(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
    elif transf == 3:
        Y_hat_df['DeepAr_og'] = np.expm1(Y_hat_df['DeepAR'])
    elif transf == 4:
        Y_hat_df['DeepAr_og'] = Y_hat_df['DeepAR']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        Y_hat_df['DeepAr_og'] = utilities.reconstruccion_diff2(Y_hat_df['DeepAR'], y_t_1, y_t_2)
    
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('DeepAr_og', 'y'), axis=1)

    preds_gen = Y_hat_df.merge(weekly_inpc, on=['unique_id', 'ds'], how='left')
    preds_gen.dropna(inplace=True)
    print(mean_absolute_error(preds_gen['y'], preds_gen['DeepAr_og']))

    # MAPE
    total_mape = comparativa['mape'].mean()
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['DeepAr_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['DeepAr_og'])
    # MSE
    mse = mean_squared_error(comparativa['y'], comparativa['DeepAr_og'])

    y_true = comparativa['y'].to_numpy()
    y_pred = comparativa['DeepAr_og'].to_numpy()
    pen_l = utilities.linearity_penalty(y_true, y_pred)  # dispara si ŷ ≈ recta

    # With Bayesian optimization we return the error, but since we maximizing
    # we have to send a negative version of it. 
    score = -(mae + 10*pen_l)
    return score

    if config['metric'] == 0:
        return -total_mape
        #tune.report({"error": mape_total})

    elif config['metric'] == 1:
        return -mae
        #tune.report({"error": mae})

    elif config['metric'] == 2:
        return -rmse
        #tune.report({"error": rmse})

    elif config['metric'] == 3:
        return -mse
        #tune.report({"error": mse})

# Transformer
def obj_transformer(data=None,
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
                    transf=None
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
            'transf':transf
            }

    # Rango
    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            #'dayofmonth': 31,
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
    # Transformaciones
    if config['transf'] == 0: # Simple Diff
        x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
        x_train = x_train.dropna()  #El primer valor será NaN

    elif config['transf'] == 1: # Logp1 Diff
        x_train['y_log'] = np.log1p(x_train['y'])
        x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
        x_train = x_train.dropna()

    elif config['transf'] == 2: # PCT
        x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
        x_train = x_train.dropna()  #El primer valor será NaN
        
    elif config['transf'] == 3: # Logp1
        x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

    elif config['transf'] == 4: # None
        x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

    elif config['transf'] == 5: # Double Diff
        x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
        x_train = x_train.dropna()

    # Estandarización
    transf = config['transf']
    mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
    sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
    x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu)/sigma


    # Escalar
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Senoidales
    senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'], 
                                                            top_k=config['signals'], 
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    config['conv_size'] = 2 ** int(config['conv_size'])

    # El horizonte siempre estará fijo, así permanece igual. 
    horizonte = int(config['h'])+ len(x_val)

    # Variables Exogenas
    if config['feats']==0: # Ninguna
        features=None
    elif config['feats']==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    elif config['feats']==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        features += list(senoidales.columns)[:-1]
    # El primer Sample d Resultados tuvo un learning rate de .0001
    # Es importante considerar ajustar este termino para futuros
    # Experiments. 
    nf = NeuralForecast(
        models=[VanillaTransformer(h=horizonte,
                                input_size=horizonte*int(config['input_size']),
                                hidden_size=config['neurons'],
                                conv_hidden_size=config['conv_size'],
                                n_head=int(config['n_heads']),
                                # Metricas de Evaluación
                                loss=metric_map[config['metric']],
                                valid_loss=metric_map[config['metric']],
                                # Escalamiento de Datos / Data Scaling
                                futr_exog_list=features,
                                scaler_type='standard',
                                learning_rate=config['learning_rate'],
                                max_steps=int(config['max_steps']),
                                val_check_steps=50,
                                early_stop_patience_steps=-1,
                                enable_progress_bar=False,
                                start_padding_enabled=False,
                                random_seed=119),
        ],
        freq=frequency_map[config['freq']]
    )
    try:
        nf.fit(df=x_train, target_col='y_scaled', verbose=False)
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

    except Exception as e:
        print(e)
        return -1000

    Y_hat_df['VanillaTransformer'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
    Y_hat_df['VanillaTransformer']  = (Y_hat_df['VanillaTransformer']*sigma)+mu

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
        Y_hat_df['Transformer_og'] = utilities.reconstruccion_diff2(Y_hat_df['VanillaTransformer'], y_t_1, y_t_2)

    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('Transformer_og', 'y'), axis=1)

    preds_gen = Y_hat_df.merge(weekly_inpc, on=['unique_id', 'ds'], how='left')
    preds_gen.dropna(inplace=True)
    print(mean_absolute_error(preds_gen['y'], preds_gen['Transformer_og']))

    # MAPE
    mape_total = comparativa['mape'].mean()
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['Transformer_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['Transformer_og'])
    # MSE
    mse = mean_squared_error(comparativa['y'], comparativa['Transformer_og'])

    y_true = comparativa['y'].to_numpy()
    y_pred = comparativa['Transformer_og'].to_numpy()
    pen_l = utilities.linearity_penalty(y_true, y_pred)  # dispara si ŷ ≈ recta

    # With Bayesian optimization we return the error, but since we maximizing
    # we have to send a negative version of it. 
    score = -(mae + 10*pen_l)
    return score

    # With Bayesian optimization we return the error, but since we maximizing
    # we have to send a negative version of it. 
    if config['metric'] == 0:
        return -mape_total
        #tune.report({"error": mape_total})

    elif config['metric'] == 1:
        return -mae
        #tune.report({"error": mae})

    elif config['metric'] == 2:
        return -rmse
        #tune.report({"error": rmse})

    elif config['metric'] == 3:
        return -mse
        #tune.report({"error": mse})

# NHITS **** Esta TRICKY. - Dado q no es parte de la investigación principal, no haré la conversión de momento
# Quizá luego aplique la conversión. 
def obj_nhits(config=None, data=None):

    config = {
        'feats':int(feats)
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
    if config['transf'] == 0: # Simple Diff
        x_train[f'y_{inverse_map_tr[0]}'] = x_train['y'].diff()
        x_train = x_train.dropna()  #El primer valor será NaN

    elif config['transf'] == 1: # Logp1 Diff
        x_train['y_log'] = np.log1p(x_train['y'])
        x_train[f'y_{inverse_map_tr[1]}'] = x_train['y_log'].diff()
        x_train = x_train.dropna()

    elif config['transf'] == 2: # PCT
        x_train[f'y_{inverse_map_tr[2]}'] = x_train['y'].pct_change() * 100
        x_train = x_train.dropna()  #El primer valor será NaN
        
    elif config['transf'] == 3: # Logp1
        x_train[f'y_{inverse_map_tr[3]}'] = np.log1p(x_train['y'])

    elif config['transf'] == 4: # None
        x_train[f'y_{inverse_map_tr[4]}'] = x_train['y']

    elif config['transf'] == 5: # Double Diff
        x_train[f'y_{inverse_map_tr[5]}'] = x_train['y'].diff().diff()
        x_train = x_train.dropna()

    # Estandarización
    transf = config['transf']
    mu = x_train[f'y_{inverse_map_tr[transf]}'].mean()
    sigma = x_train[f'y_{inverse_map_tr[transf]}'].std()
    x_train['y_estandarizada'] = (x_train[f'y_{inverse_map_tr[transf]}'] - mu)/sigma

    # Escalar
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    
    # Senoidales
    senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'], 
                                                            top_k=4,
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Conversion - Neuronal
    config['neurons'] = 2 ** int(config['neurons'])
    horizonte = config['h'] + len(x_val)

    # Variables Exogenas
    if config['feats']==0: # Ninguna
        features=None
    elif config['feats']==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos']]
    elif config['feats']==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos']]
        features += list(senoidales.columns)

    nf = NeuralForecast(
        models=[NHITS(h=horizonte,
                    input_size=horizonte*int(config['input_size']),
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
                    max_steps=int(config['max_steps']),
                    val_check_steps=50,
                    early_stop_patience_steps=-1,
                    enable_progress_bar=False,
                    start_padding_enabled=True,
                    random_seed=119
                    ),
        ],
        freq=frequency_map[config['freq']]
    )
    nf.fit(df=x_train, target_col='y_scaled', verbose=False)
    Y_hat_df = nf.predict(verbose=False)
    Y_hat_df['NHITS'] = scaler.inverse_transform([Y_hat_df['NHITS'].values])[0]
    Y_hat_df['NHITS']  = (Y_hat_df['NHITS']*sigma)+mu

    # Reconstruccion
    transf = config['transf']
    if transf == 0:
        Y_hat_df['nhits_og'] = utilities.reconstruccion_diff(Y_hat_df['NHITS'], x_train['y'].iloc[-1])
    elif transf == 1:
        Y_hat_df['nhits_og'] = utilities.reconstruccion_log_diff(Y_hat_df['NHITS'], x_train['y'].iloc[-1])
    elif transf == 2:
        Y_hat_df['nhits_og'] = utilities.reconstruccion_pct(Y_hat_df['NHITS'], x_train['y'].iloc[-1])
    elif transf == 3:
        Y_hat_df['nhits_og'] = np.expm1(Y_hat_df['NHITS'])
    elif transf == 4:
        Y_hat_df['nhits_og'] = Y_hat_df['NHITS']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        Y_hat_df['nhits_og'] = utilities.reconstruccion_diff2(Y_hat_df['NHITS'], y_t_1, y_t_2)

    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    # MAE
    mae = mean_absolute_error(comparativa['y'], comparativa['nhits_og'])
    # RMSE
    rmse = root_mean_squared_error(comparativa['y'], comparativa['nhits_og'])
    # MAPE
    comparativa['mape'] = comparativa.apply(accuracy.mape, args=('nhits_og', 'y'), axis=1)
    total_mape = comparativa['mape'].mean()
    # MSE
    mse = mean_squared_error(comparativa['y'], comparativa['nhits_og'])

    # With Bayesian optimization we return the error, but since we maximizing
    # we have to send a negative version of it. 
    if config['metric'] == 0:
        return -total_mape
        #tune.report({"error": mape_total})

    elif config['metric'] == 1:
        return -mae
        #tune.report({"error": mae})

    elif config['metric'] == 2:
        return -rmse
        #tune.report({"error": rmse})

    elif config['metric'] == 3:
        return -mse
        #tune.report({"error": mse})

# Candidatos a ser Agregados. #NBEATS no lo sé. En teoría, NHITs es el sucesor.

# Posiblemente ~ Informer

# Y Patch TFT

# Temporal Fusion Transformer

# DVAE




