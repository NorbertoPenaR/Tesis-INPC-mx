#Predict Bayes

# Predict
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from utiles import utilities
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from hyperopt import hp
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR, VanillaTransformer, LSTM, RNN, NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE, MSE

from sklearn.preprocessing import MinMaxScaler
import accuracy

# Modelos Clasicos, BenchMarks. 
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage, Naive, RandomWalkWithDrift

# DVAE modules.
import yaml
from dvae_v2.train import main as train_main
from dvae_v2.predict import main as predict_main

# XGBRegressor. Lag Features and More.
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, ExpandingMean
from mlforecast.target_transforms import Differences
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from mlforecast.target_transforms import GlobalSklearnTransformer
sk_log1p = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

# Mapeo de Metricas
metric_map_number_fun = {
    1: MAE(),
    0: MAPE(),
    2: RMSE(),
    3: MSE()
}

metric_map = {
    0: MAPE(),
    1: MAE(),
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

preds=[]

# Just cheking IN.
simulation_dates = utilities.ultimos_dias_meses(n=6, frecuencia=3, referencia='2025-01-01')
import process_data
inpc_path_Q = 'ca56_2018a.csv'
inpc_Q = process_data.limpiar_csv_inegi(inpc_path_Q)
weekly_inpc = process_data.inpc_data_weekly(datos=inpc_Q)

inpc_path_M = 'ca55_2018a.csv'
inpc_M = process_data.limpiar_csv_inegi(inpc_path_M)
monthly_inpc = process_data.inpc_monthly(datos=inpc_M)

def predict_avg_naive_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                        train_years=int(config['years']), 
                        months_val=int(config['months']), 
                        date='ds')
    
    whole = pd.concat([x_train, x_val])
    
    avg_method = HistoricAverage()
    naive_method = Naive()
    drift_method = RandomWalkWithDrift()
    sf = StatsForecast(models=[drift_method, avg_method, naive_method], 
                       freq=frequency_map[config['freq']],)
    sf.fit(whole)

    fcasts = sf.forecast(df=whole, h=int(config['h']), level=[95])
    dff = fcasts.merge(unseen, on=['unique_id', 'ds'], how='inner')

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
def predict_holt_winters_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    x_train_val = pd.concat([x_train, x_val])
    x_train = x_train_val.copy()
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
    H['holt_w'] = HoltWinters.forecast(steps=int(config['h']))

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
def predict_xgb_cv(config=None, data=None, cutoff_date=None):
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
    for col, max_val in cyclic_cols.items():
        data = utilities.add_cyclic_features(data, col, max_val)

    # División de Conjuntos.
    x_train, x_val = utilities.split_data_val(data=data, 
                                            train_years=int(config['years']), 
                                            months_val=int(config['months']), 
                                            date='ds')
    
    x_train_val = pd.concat([x_train, x_val])

    xgb_exo = MLForecast(
                models=XGBRegressor(
                    n_estimators=int(config['num_boost_round']), # 200 seems to work fine
                    max_depth=int(config['max_depth']),
                    learning_rate=config['eta'],
                    subsample=config['subsample'],
                    colsample_bytree=config['colsample_bytree'],
                    random_state=119,
                    reg_alpha=config['alpha'],
                    reg_lambda=config['lambdaa']
                ),
                freq=frequency_map[config['freq']],
                lags=list(range(1,int(52*config['input_mult']+1))),
                #lags=list(range(1,51)),
                target_transforms=[Differences([1])],
            )
    
    # XGBoost only has 2 options, unless lags are added.
    if int(config['feats'])==0:
        xgb_exo.fit(x_train_val[['ds', 'unique_id', 'y']])
        dff = xgb_exo.predict(h=int(config['h']))

    elif int(config['feats'])==1:
        feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        xgb_exo.fit(x_train[['ds', 'unique_id', 'y']+feats], static_features=[])

        future_df = xgb_exo.make_future_dataframe(h=int(config['h']))
        future_df = utilities.features_from_date(future_df, 'ds')
        for col, max_val in cyclic_cols.items():
            future_df = utilities.add_cyclic_features(future_df, col, max_val)
        
        dff = xgb_exo.predict(h=int(config['h']), X_df=future_df)
        

    elif int(config['feats'])==2:# Features Senoidales added.
        # Senoidales
        senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train_val['y'],
                                                                top_k=int(config['signals']),
                                                                extra_steps=int(config['h']))
        # Entrenamiento
        for c in senoidales.columns:
            x_train_val[c] = (senoidales[c]).values[:len(x_train_val)]
        
        feats = [f"{col}_{trig}" for col in cyclic_cols.keys() for trig in ['cos', 'sin']]
        
        feats += list(senoidales.columns)[:-1]
        xgb_exo.fit(x_train_val[['ds', 'unique_id', 'y']+feats], static_features=[])                
        future_df = xgb_exo.make_future_dataframe(h=int(config['h']))
        future_df = utilities.features_from_date(future_df, 'ds')

        for col, max_val in cyclic_cols.items(): # Temporales
            future_df = utilities.add_cyclic_features(future_df, col, max_val)
        for c in senoidales.columns: # Fast Fourier Transform 
            #x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))]
            future_df[c] = (senoidales[c]).values[len(x_train_val):]

        dff = xgb_exo.predict(h=int(config['h']), X_df=future_df)

    #print('Preds')
    #print(dff)

    # Merge con los resultados
    results = dff.merge(unseen, on=['ds', 'unique_id'], how='inner')
    results['mape'] = results.apply(accuracy.mape, args=('XGBRegressor', 'y'), axis=1)
    results['diff'] = abs(results['y'] - results['XGBRegressor'])
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, results])
    #print(performance)

    return performance, dff, results

# Neural Network Models. 

# RNN
def predict_rnn_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

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
    cv_preds = []
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
                                                                extra_steps=int(config['h']))
        for c in senoidales.columns:
            x_train[c] = (senoidales[c]).values[:len(x_train)]

        # Horizonte
        horizonte = int(config['h'])

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
        
        Y_hat_df['cutoff_cv'] = dt_cv 
        cv_preds.append(Y_hat_df[['ds', 'unique_id', 'cutoff_cv', 'rnn_og']])

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
        #mse = mean_squared_error(comparativa['y'], comparativa['rnn_og'])
        #cv_mse.append(mse)

    print(cv_preds[0])
    performance_cv = pd.concat(cv_preds)
    performance_cv = x_train_val[['ds', 'unique_id', 'y']].merge(performance_cv, 
                                    on=['ds', 'unique_id'],
                                    how='left')
    
    print(performance_cv.ds.min())
    print(performance_cv.ds.max())
    print(performance_cv)

    # Se junta el conjunto de validación para formar el gran conjunto, el cual será dividido en 4 splits con horizontes 
    # Previamente definidos por el usuario desde el modulo main.py 
    #x_train_val = pd.concat([x_train, x_val])
    x_train = x_train_val.copy()
    print(x_train.head())
    print(x_train.tail())
    
    # Conversión de Neuronas.
    #config['neurons'] = 2 ** int(config['neurons'])
    # Conversión Batch Size
    #config['batch_size'] = 2 ** int(config['batch_size'])

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
                                                            extra_steps=int(config['h']))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Horizonte
    horizonte = int(config['h'])
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
        models=[RNN(h=int(config['h']),
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

    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['rnn_og'])
    results['mape'] = results.apply(accuracy.mape, args=('rnn_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance_cv = pd.concat([performance_cv, results])
    return performance_cv, Y_hat_df, results

# LSTM
def predict_lstm_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

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
    x_train = x_train_val.copy()
    
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    #simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())
    print(config)
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size']) # 64, 128 
    print(config)
    
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
                                                            top_k=int(config['signals']),
                                                            extra_steps=int(config['h']))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Horizonte
    horizonte = int(config['h'])

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
                        input_size=int(horizonte*config['input_size']),
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
                        random_seed=119
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
    
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['lstm_og'])
    results['mape'] = results.apply(accuracy.mape, args=('lstm_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, results])
    return performance, Y_hat_df, results
    results.to_csv('lstm_prediction.csv')

# DeepAr
def predict_deepAr_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]    
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

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
    x_train = x_train_val.copy()
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    #simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size'])

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
                                                            extra_steps=int(config['h']))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Horizonte
    horizonte = int(config['h'])

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
    
    # Resultados    
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['deepAr_og'])
    results['mape'] = results.apply(accuracy.mape, args=('deepAr_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, results])
    return performance, Y_hat_df, results
    results.to_csv('deep_ar_prediction.csv')

# Transformer
def predict_transformer_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

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
    x_train = x_train_val.copy()
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())
    
    # Conversión de Neuronas. 
    config['neurons'] = 2 ** int(config['neurons'])
    config['conv_size'] = 2 ** int(config['conv_size'])
    # Conversión Batch Size
    config['batch_size'] = 2 ** int(config['batch_size'])

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
                                                            extra_steps=int(config['h']))
    for c in senoidales.columns:
        x_train[c] = (senoidales[c]).values[:len(x_train)]

    # Horizonte
    horizonte = int(config['h'])

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
                                input_size=int(horizonte*config['input_size']),
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
                                                                    y_t_1, 
                                                                    y_t_2)
        
    # Val set. 
    #comparativa = Y_hat_df.merge(x_val, on=['unique_id', 'ds'], how='inner')
    # Resultados
    results = Y_hat_df.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['Transformer_og'])
    results['mape'] = results.apply(accuracy.mape, args=('Transformer_og', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, results])
    return performance, Y_hat_df, results

def predict_d3vae_cv(config=None, data=None, cutoff_date=None):
    unseen = data[data['ds']>cutoff_date]
    data = data[data['ds']<=cutoff_date]
    data['ds'] = pd.to_datetime(data['ds'])

    # Conversión de Neuronas.
    config['neurons'] = 2 ** int(config['neurons'])
    # conversión de Batch size.
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
    x_train = x_train_val.copy()
    str_datee = x_train_val.ds.max()
    str_datee = str(str_datee)[: 10]
    
    # Se obtienen las fechas que serviran para cortar el conjunto para validación cruzada.
    #simulation_dates_cv = utilities.ultimos_dias_meses(n=4, frecuencia=12, referencia=x_train_val.ds.max())

    if config['feats'] == 0:
        use_time_features=False
        use_fft_features=False

    elif config['feats'] == 1:
        use_time_features=True
        use_fft_features=False

    elif config['feats'] == 2:
        use_time_features=True
        use_fft_features=True

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
    
    x_train[['unique_id', 'ds', 'y_estandarizada']].rename(
            columns={'y_estandarizada':'y'}
        ).to_csv('temporal_file_d3vae.csv')
    
    #x_train.to_csv('temporal_file_d3vae.csv')
    
    config_d3vae = {
        'data': {
            'csv_path': 'temporal_file_d3vae.csv',
            'context_len': int(config['h']*config['input_size']),
            'horizon': int(config['h']),
            'stride': 1,
            'batch_size': int(config['batch_size']),
            'num_workers': 0,
            'val_split': 0.2,
            'normalize': 'zscore',
            'models_to_train': ["d3vae"], ## Para determinar el 
            'minmax_range': [0.0, 1.0],
            'time_features': ["dow", "month", "weekofyear", "dayofyear"],   # produce sin/cos por cada una "dow", 
            'use_time_features': use_time_features,
            'use_fourier': False,
            'use_fft_features': use_fft_features,
            'fft_top_k': int(config['signals']),
            'fft_scale_range': [0.0, 1.0], # escala de las ondas
            'fft_include_signal': False,# agrega la señal reconstruida como extra, en general, es mejor no incluirla por que 
                                        # Leakea información pasada, y hace que los modelos aprendan a seguir la señal, la cual
                                        # Será erronea a futuro. Es mejor aprendan sin la señal. 
            'fft_window': "hann",       # opcional: "hann" o null, hann por default.
            'freq': frequency_map[config['freq']], # El sistema puede inferirla, pero es mejor definirla.
            'split_mode': 'time',   #  holdout temporal por serie
            'cutoff': str_datee, #'2023-09-18',
            'from_date': '2000-01-01',
            'id': 'Inflacion'
        },

        'd3vae': {
            'T': 50,
            'schedule': "cosine",         # o "cosine" / "linear"
            'beta_x': [1e-4, 2e-2],
            'beta_y': [1e-4, 2e-2],
            'time_emb_dim': 32+12,
            'dsm_weight': 0.2,
            'tc_weight': 0.1,
            'jump_gamma': 0.0,
            'jump_t': 0,
        },

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
            'kl_warmup_epochs': 15, # estaba en 7
            'teacher_forcing_start': 0.7,
            'teacher_forcing_end': 0.1,
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

    # Resultados
    results = preds.merge(unseen, on=['unique_id', 'ds'], how='inner')
    results['diff'] = abs(results['y'] - results['d3vae'])
    results['mape'] = results.apply(accuracy.mape, args=('d3vae', 'y'), axis=1)
    results['acc'] = round(100 - results['mape'] , 4)
    performance = pd.concat([x_train, results])
    return performance, preds, results

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