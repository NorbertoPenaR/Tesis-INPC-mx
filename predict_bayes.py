#Predict Bayes

# Predict
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from utiles import utilities
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, VanillaTransformer, LSTM, RNN, NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE, MSE
import tensorflow as tf
from ray import tune
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import accuracy

from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage, Naive, RandomWalkWithDrift

# Mapeo de Metricas
metric_map = {
    1: MAE(),
    0: MAPE(),
    2: RMSE(),
    3: MSE()
}
# Mapeo Frequencias
frequency_map = {
    0:'W-mon',
    1:'ME',
    2:'B'
}

inverse_map_tr = {
    0:'diff',
    1:'diff_logp1',
    2:'pct',
    3:'logp1',
    4:'none',
    5:'diff2'
}

# Just cheking IN.
import process_data
inpc_path_Q = 'ca56_2018a.csv'
inpc_Q = process_data.limpiar_csv_inegi(inpc_path_Q)
weekly_inpc = process_data.inpc_data_weekly(datos=inpc_Q)

def predict_avg_naive(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=10, 
                                            months_val=3, 
                                            date='ds')
    
    whole = pd.concat([x_train, x_val])
    
    avg_method = HistoricAverage()
    naive_method = Naive()
    drift_method = RandomWalkWithDrift()
    sf = StatsForecast(models=[drift_method, avg_method, naive_method], freq=config['freq'])
    sf.fit(whole)

    fcasts = sf.forecast(df=whole, h=config['h'], level=[95])
    dff = fcasts.merge(unseen, on=['unique_id', 'ds'], how='inner')

    '''dff = pd.DataFrame(pd.date_range(start= whole['ds'].max(),
                                    periods= config['h']+1,
                                    freq=frequency_map[config['freq']]),
                                    columns=['ds'])
    dff['avg'] = whole['y'].mean()
    dff['naive'] = x_val['y'].iloc[-1]
    dff = dff[1:]'''

    return fcasts, dff

# Clasical - Discrete Fourier Transform 
def predict_dft(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

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
    
    # Id 
    id_fft = list(x_train.unique_id.unique())[0]
    
    # Escalado
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train['y_scaled'] = scaler.fit_transform(x_train[f'y_{inverse_map_tr[transf]}'].values.reshape(-1, 1))

    senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_scaled'],
                                                            top_k=int(config['signals']),
                                                            extra_steps=config['h']+ len(x_val))
    
    dff = pd.DataFrame(pd.date_range(start= x_val['ds'].min(),
                                    periods= config['h']+ len(x_val),
                                    freq=frequency_map[config['freq']]),
                                    columns=['ds'])
    
    for c in senoidales.columns:
        dff[c] = (senoidales[c]).values[len(x_train):]
    dff['signal'] = scaler.inverse_transform([dff['signal'].values])[0]

    transf = config['transf']
    if transf == 0:
        dff['yhat_og'] = utilities.reconstruccion_diff(dff['signal'], x_train['y'].iloc[-1])
    elif transf == 1:
        dff['yhat_og'] = utilities.reconstruccion_log_diff(dff['signal'], x_train['y'].iloc[-1])
    elif transf == 2:
        dff['yhat_og'] = utilities.reconstruccion_pct(dff['signal'], x_train['y'].iloc[-1])
    elif transf == 3:
        dff['yhat_og'] = np.expm1(dff['signal'])
    elif transf == 4:
        dff['yhat_og'] = dff['signal']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        dff['yhat_og'] = utilities.reconstruccion_diff2(dff['signal'], y_t_1, y_t_2)

    dff['unique_id'] = id_fft # unique id as it is xd. 119

    comparativa = dff.merge(x_val[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')

    # Merge con los resultados
    results = dff.merge(unseen, on=['ds', 'unique_id'], how='inner')
    #print('Results')
    #print(results)

    results['mape'] = results.apply(accuracy.mape, args=('yhat_og', 'y'), axis=1)
    results['diff'] = abs(results['y'] - results['yhat_og'])
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, comparativa, results])
    #print(performance)

    return performance, dff, results


# Holt Winters
def predict_holt_winters(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    # Holt-Winters Parameters - Convertion
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
    scaler = MinMaxScaler(feature_range=(1, 10))
    x_train['y_scaled'] = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))

    # Id 
    id_holt = list(x_train.unique_id.unique())[0]

    #mu = x_train['y'].mean()
    #sigma = x_train['y'].std()
    #x_train['y_estandarizada'] = (x_train['y']-mu)/sigma
    #scaler = MinMaxScaler(feature_range=(1, 10))
    #data_scaled = scaler.fit_transform(x_train['y_estandarizada'].values.reshape(-1, 1))
    #x_train['y_scaled'] = data_scaled
    #features_df, t_total, info_frec = utilities.generar_senoidales_exogenas(x_train['y_scaled'], top_k=4, extra_steps=config['h'])
    
    HoltWinters = ExponentialSmoothing(x_train['y_scaled'],
                    dates=x_train['ds'],
                    trend=config['trend_type'],  # 'add'
                    seasonal=config['seasonal_type'],  # 'add', 'mul'
                    seasonal_periods= int(config['seasonal_periods']),#int(info_frec[1]['periodo']),
                    damped_trend=config['damped_trend'],  # True / False
                    use_boxcox=config['use_boxcox'],
                    freq=frequency_map[config['freq']]
                    ).fit()
    
    #forecast = HoltWinters.forecast(steps=len(x_val))
    # Predicciones
    H = pd.DataFrame()
    H['holt_w'] = HoltWinters.forecast(steps=int(config['h']+ len(x_val)))

    # Paso 4.1: Invertir escalado
    H['holt_w_v2'] = scaler.inverse_transform(H[['holt_w']])

    # Paso 4.2: Invertir estandarización
    H['Holt'] = H['holt_w_v2'] * sigma + mu

    # Paso 4.3: Reconstruir serie original desde el último valor real
    transf = config['transf']
    if transf == 0:
        H['holt_winters_og'] = utilities.reconstruccion_diff(H['Holt'], x_train['y'].iloc[-1])
    elif transf == 1:
        H['holt_winters_og'] = utilities.reconstruccion_log_diff(H['Holt'], x_train['y'].iloc[-1])
    elif transf == 2:
        H['holt_winters_og'] = utilities.reconstruccion_pct(H['Holt'], x_train['y'].iloc[-1])
    elif transf == 3:
        H['holt_winters_og'] = np.expm1(H['Holt'])
    elif transf == 4:
        H['holt_winters_og'] = H['Holt']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        H['holt_winters_og'] = utilities.reconstruccion_diff2(H['Holt'], y_t_1, y_t_2)

    H = H.reset_index().rename(columns={'index':'ds'})
    H['unique_id'] = id_holt

    comparativa = H.merge(x_val[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')

    results = H.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['holt_winters_og'])
    results['mape'] = results.apply(accuracy.mape, args=('holt_winters_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    
    return results, H
# Tree Methods - ML

# XGB 
def predict_xgb(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    # We sent it to 1, so we can get past, but does not matter if we are doing weekly. 
    # We can adapt it tho. 
    #unseen['ds'] = unseen['ds'].apply(lambda x: x.replace(day=1))

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

    transf = config['transf']
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
    
    # ID.
    id_xgb = list(x_train.unique_id.unique())[0]

    # XGBoost only has 2 options, unless lags are added.
    if int(config['feats'])==0 or int(config['feats'])==1:
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    elif int(config['feats'])==2: # Temporales + Senoidales
        # Features Senoidales added.
        # Senoidales
        senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train[f'y_{inverse_map_tr[transf]}'],
                                                                top_k=int(config['signals']), 
                                                                extra_steps=config['h']+ len(x_val))
        # Entrenamiento
        for c in senoidales.columns:
            x_train[c] = (senoidales[c]).values[:len(x_train)]
        # Validación
        for c in senoidales.columns:
            x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))]

        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        features += list(senoidales.columns)[:-1]
    
    # Matrices de XGBoost
    dtrain = xgb.DMatrix(x_train[features], label=x_train[f'y_{inverse_map_tr[transf]}'], feature_names=features)
    dval = xgb.DMatrix(x_val[features], label=x_val[f'y_{inverse_map_tr[transf]}'], feature_names=features)

    # Matriz con toda la historia
    param = {
        'max_depth': int(config['max_depth']),
        'colsample_bytree': config['colsample_bytree'],
        'subsample': config['subsample'],
        'seed': 0,
        'verbosity': 0,
        'alpha': config['alpha'],
        'eta': config['eta'],
        'lambda': config['lambdaa'],
        'tree_method': 'hist',
        'gamma':.1,
        'max_bin':256,
        'eval_metric':"mae"#'mape'
    }
    
    # Train XGBoost model on training set
    xgb_model_train = xgb.train(
        param,
        dtrain,
        num_boost_round=int(config['num_boost_round']),
        early_stopping_rounds=50,
        verbose_eval=False,
        evals=[(dval, 'val')])
    
    dff = pd.DataFrame(pd.date_range(start= x_val['ds'].min(),
                                    periods= config['h']+ len(x_val),
                                    freq=frequency_map[config['freq']]),
                                    columns=['ds'])
    
    # Should be applied only with clima
    #dff['ds'] = dff['ds'].apply(lambda x: x.replace(day=1))
    
    dff = utilities.features_from_date(dff, 'ds')
    for col, max_val in cyclic_cols.items():
        dff = utilities.add_cyclic_features(dff, col, max_val)
    
    if int(config['feats'])==2: 
        for c in senoidales.columns:
            dff[c] = (senoidales[c]).values[len(x_train):]
    
    dff_matrix = xgb.DMatrix(dff[features], feature_names=features)
    dff['xgb'] = xgb_model_train.predict(dff_matrix) # Prediccion

    # Reconstruccion
    if transf == 0:
        dff['xgb_og'] = utilities.reconstruccion_diff(dff['xgb'], x_train['y'].iloc[-1])
    elif transf == 1:
        dff['xgb_og'] = utilities.reconstruccion_log_diff(dff['xgb'], x_train['y'].iloc[-1])
    elif transf == 2:
        dff['xgb_og'] = utilities.reconstruccion_pct(dff['xgb'], x_train['y'].iloc[-1])
    elif transf == 3:
        dff['xgb_og'] = np.expm1(dff['xgb'])
    elif transf == 4:
        dff['xgb_og'] = dff['xgb']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        dff['xgb_og'] = utilities.reconstruccion_diff2(dff['xgb'], y_t_1, y_t_2)

    dff['unique_id'] = id_xgb # unique id as it is xd. 119

    comparativa = dff.merge(x_val[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')

    # Merge con los resultados
    results = dff.merge(unseen, on=['ds', 'unique_id'], how='inner')
    #print('Results')
    #print(results)

    results['mape'] = results.apply(accuracy.mape, args=('xgb_og', 'y'), axis=1)
    results['diff'] = abs(results['y'] - results['xgb_og'])
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, comparativa, results])
    #print(performance)

    return performance, dff, results

# Neural Network Models. 

# RNN
def predict_rnn(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    # Variables de Tiempo
    data = utilities.features_from_date(data, 'ds')
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
                                                            top_k=int(config['signals']), 
                                                            extra_steps= config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Variables Exogenas
    if int(config['feats'])==0: # Ninguna
        features=None
    elif int(config['feats'])==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    elif int(config['feats'])==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        features += list(senoidales.columns)[:-1]

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    horizonte = int(config['h']) + len(x_val)
    nf = NeuralForecast(
        models=[RNN(h=horizonte,
                    input_size=horizonte*int(config['input_size']),
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
    # It trains the model
    nf.fit(df=x_train, target_col='y_scaled')
    
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
        Y_hat_df['rnn_og'] = utilities.reconstruccion_diff2(Y_hat_df['RNN'], 
                                                            y_t_1, 
                                                            y_t_2)
    
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['rnn_og'])
    results['mape'] = results.apply(accuracy.mape, args=('rnn_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, comparativa, results])
    return performance, Y_hat_df, results

# LSTM
def predict_lstm(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    cyclic_cols = {
        'month': 12,
        'weekofyear': 52,
        'dayofyear': 366,
        'dayofmonth': 31,
        'quarter': 4,
        }

    data = utilities.features_from_date(data, 'ds')
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

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
                                                            top_k=int(config['signals']), 
                                                            extra_steps= config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Variables Exogenas
    if int(config['feats'])==0: # Ninguna
        features=None
    elif int(config['feats'])==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    elif int(config['feats'])==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        features += list(senoidales.columns)[:-1]

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Horizonte
    horizonte = int(config['h']) + len(x_val)
    nf = NeuralForecast(
        models=[LSTM(h=horizonte,
                    #input_size=24,
                    input_size=horizonte*int(config['input_size']),
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
                    # Featrues 
                    futr_exog_list=features,
                    max_steps=int(config['max_steps']),
                    scaler_type='standard',
                    enable_progress_bar=False,
                    #start_padding_enabled=True
                    random_seed=119
                    ),
        ],
        freq=frequency_map[config['freq']]
    )

    # It trains the model
    nf.fit(df=x_train, target_col='y_scaled')

    if int(config['feats'])==0: # Ninguna
        Y_hat_df = nf.predict(verbose=0)

    elif int(config['feats'])==1: # Temporales
        df_features = nf.make_future_dataframe()
        df_features = utilities.features_from_date(df_features, 'ds')
        for col, max_val in cyclic_cols.items():
            df_features = utilities.add_cyclic_features(df_features, col, max_val)
        Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

    elif int(config['feats'])==2: # Temporales + Senoidales
        df_features = nf.make_future_dataframe()
        df_features = utilities.features_from_date(df_features, 'ds')
        for col, max_val in cyclic_cols.items():
            df_features = utilities.add_cyclic_features(df_features, col, max_val)
        for c in senoidales.columns:
            df_features[c] = (senoidales[c]).values[len(x_train):]
        Y_hat_df = nf.predict(futr_df= df_features, verbose=0)

    Y_hat_df['LSTM'] = scaler.inverse_transform([Y_hat_df['LSTM'].values])[0]
    # Desestandarización. I dont think its a good thing. Nor a bad one. 
    #Y_hat_df['LSTM']  = (Y_hat_df['LSTM']*sigma)+mu

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
        Y_hat_df['lstm_og'] = utilities.reconstruccion_diff2(Y_hat_df['LSTM'], 
                                                            y_t_1, 
                                                            y_t_2)

    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['lstm_og'])
    results['mape'] = results.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, comparativa, results])
    return performance, Y_hat_df, results
    results.to_csv('lstm_prediction.csv')

# DeepAr
def predict_deepAr(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]    
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    data = utilities.features_from_date(data, 'ds')
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

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
    senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train['y'], #x_train[f'y_{inverse_map_tr[transf]}'], 
                                                            top_k=int(config['signals']), 
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Variables Exogenas
    if int(config['feats'])==0: # Ninguna
        features=None
    elif int(config['feats'])==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    elif int(config['feats'])==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        features += list(senoidales.columns)[:-1]

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    horizonte = int(config['h']) + len(x_val)
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
                    futr_exog_list=features,
                    max_steps=int(config['max_steps']),
                    val_check_steps=100,
                    early_stop_patience_steps=-1,
                    scaler_type='identity',
                    enable_progress_bar=False,
                    random_seed=119,
                    #start_padding_enabled=True
                    ),
        ],
        freq=frequency_map[config['freq']]
    )

    # It trains the model
    nf.fit(df=x_train, target_col='y_scaled')
    
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

    Y_hat_df['DeepAR'] = scaler.inverse_transform([Y_hat_df['DeepAR'].values])[0]
    Y_hat_df['DeepAR']  = (Y_hat_df['DeepAR']*sigma)+mu

    # Reconstruccion
    transf = config['transf']
    if transf == 0:
        Y_hat_df['deepAr_og'] = utilities.reconstruccion_diff(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
    elif transf == 1:
        Y_hat_df['deepAr_og'] = utilities.reconstruccion_log_diff(Y_hat_df['DeepAR'], x_train['y'].iloc[-1])
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
    
    # Resultados
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['deepAr_og'])
    results['mape'] = results.apply(accuracy.mape, args=('deepAr_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, comparativa, results])
    return performance, Y_hat_df, results
    results.to_csv('deep_ar_prediction.csv')

# Transformer
def predict_transformer(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            #'dayofmonth': 31,
            'quarter': 4,
            }

    data = utilities.features_from_date(data, 'ds')
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

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
                                                            top_k=int(config['signals']), 
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Variables Exogenas
    if int(config['feats'])==0: # Ninguna
        features=None
    elif int(config['feats'])==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
    elif int(config['feats'])==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        features += list(senoidales.columns)[:-1]

    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    config['conv_size'] = 2 ** int(config['conv_size'])
    # El horizonte siempre estará fijo, así permanece igual. 
    horizonte = int(config['h']) + len(x_val)
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
                                learning_rate=config['learning_rate'],
                                futr_exog_list=features,
                                scaler_type='standard',
                                max_steps=int(config['max_steps']),
                                val_check_steps=50,
                                early_stop_patience_steps=-1,
                                enable_progress_bar=False,
                                start_padding_enabled=False,
                                random_seed=119),
        ],
        freq=frequency_map[config['freq']]
    )

    # It trains the model
    nf.fit(df=x_train, target_col='y_scaled')
    
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

    Y_hat_df['VanillaTransformer'] = scaler.inverse_transform([Y_hat_df['VanillaTransformer'].values])[0]
    Y_hat_df['VanillaTransformer'] = (Y_hat_df['VanillaTransformer']*sigma) + mu
    
    # Reconstruccion
    transf = config['transf']
    if transf == 0:
        Y_hat_df['transformer_og'] = utilities.reconstruccion_diff(Y_hat_df['VanillaTransformer'], x_train['y'].iloc[-1])
    elif transf == 1:
        Y_hat_df['transformer_og'] = utilities.reconstruccion_log_diff(Y_hat_df['VanillaTransformer'], x_train['y'].iloc[-1])
    elif transf == 2:
        Y_hat_df['transformer_og'] = utilities.reconstruccion_pct(Y_hat_df['VanillaTransformer'], x_train['y'].iloc[-1])
    elif transf == 3:
        Y_hat_df['transformer_og'] = np.expm1(Y_hat_df['VanillaTransformer'])
    elif transf == 4:
        Y_hat_df['transformer_og'] = Y_hat_df['VanillaTransformer']
    elif transf == 5:
        y_t_1 = x_train['y'].iloc[-1]
        y_t_2 = x_train['y'].iloc[-2]
        Y_hat_df['transformer_og'] = utilities.reconstruccion_diff2(Y_hat_df['VanillaTransformer'], 
                                                                    y_t_1, 
                                                                    y_t_2)
    # Resultados
    comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    try:
            
        preds_gen = Y_hat_df.merge(weekly_inpc, on=['unique_id', 'ds'], how='left')
        preds_gen.dropna(inplace=True)
        print(mean_absolute_error(preds_gen['y'], preds_gen['transformer_og']))
    except Exception as e:
        print(e)
        print(preds_gen)
    
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['transformer_og'])
    results['mape'] = results.apply(accuracy.mape, args=('transformer_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, comparativa, results])
    return performance, Y_hat_df, results

# NHITS
def predict_nhits(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    cyclic_cols = {
            'month': 12,
            'weekofyear': 52,
            'dayofyear': 366,
            'dayofmonth': 31,
            'quarter': 4,
            }

    data = utilities.features_from_date(data, 'ds')
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

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
                                                            top_k=int(config['signals']), 
                                                            extra_steps=config['h']+ len(x_val))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]
    
    # Variables Exogenas
    if int(config['feats'])==0: # Ninguna
        features=None
    elif int(config['feats'])==1: # Temporales
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos']]
    elif int(config['feats'])==2: # Temporales + Senoidales
        # Features Senoidales added.
        features = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos']]
        features += list(senoidales.columns)

    config['neurons'] = 2 ** int(config['neurons'])
    horizonte = int(config['h']) + len(x_val)
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
    # It trains the model
    nf.fit(df=x_train, target_col='y_scaled')
    
    df_features = nf.make_future_dataframe()
    df_features = utilities.features_from_date(df_features, 'ds')
    for col, max_val in cyclic_cols.items():
        df_features = utilities.add_cyclic_features(df_features, col, max_val)

    for c in senoidales.columns:
        df_features[c] = (senoidales[c]).values[len(x_train):]

    Y_hat_df = nf.predict(futr_df= df_features, verbose=0)
    Y_hat_df['NHITS'] = scaler.inverse_transform([Y_hat_df['NHITS'].values])[0]
    Y_hat_df['NHITS'] = (Y_hat_df['NHITS']*sigma) + mu
    start_value = x_train['y'].iloc[-1]
    Y_hat_df['nhits_og'] = utilities.reconstruccion_pct(Y_hat_df['NHITS'], start_value)

    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['nhits_og'])
    results['mape'] = results.apply(accuracy.mape, args=('nhits_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    return results, Y_hat_df
    results.to_csv('nhits_prediction.csv')