# XGBoost
from utiles import utilities
import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
from functools import partial
from bayes_opt import BayesianOptimization
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

class xgb_senoidales:

    ds = 'ds'
    id = 'sku'
     
    def predict_xgb_month(self, data=None, category=None, dic_params=None, cutoff_date=None):
        try:
            features = ['quarter', 'month', 'year', 'dayofyear', 'weekofyear']

            q = dic_params
            data['ds'] = pd.to_datetime(data['ds'])
            data = data[data['ds']<cutoff_date]
            data['ds'] = data['ds'].apply(lambda x: x.replace(day=1))
            #data['ds'] = pd.to_datetime(data['ds'])
            #data_monthly =  data.set_index('ds').groupby(['sku', 
            #                                            'familia', 
            #                                            'CLASS_ABC']).resample('ME')['y'].sum().reset_index()
            #data_monthly['ds'] = data_monthly['ds'].apply(lambda x: x.replace(day=1))
            #data_sku = data_monthly[data_monthly['sku']==category]
            #print('Hola1')

            x_train, x_val = utilities.split_data_val(
                                data=data,
                                train_years=int(q['years']),
                                months_val=int(q['months']),
                                date='ds')
            
            x_train = utilities.laggin_features(2, 36, x_train, 'ds')
            x_train = utilities.features_from_date(x_train, 'ds')
            x_val = utilities.lags_future_features(dff=x_val.rename(columns={'y':'target'}), 
                                                    inference_data=x_train,
                                                    n_lags_start= 2, 
                                                    n_lags_end= 36,
                                                    ds='ds')

            x_val = utilities.features_from_date(x_val, 'ds')
            
            historicos = (
                x_train
                .groupby('month')['y']
                .agg(Promedio_Historico='mean', Std_Historico='std')
                .reset_index())
            x_val = x_val.merge(historicos, on=['month'], how='inner')

            # Senoidales
            senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train['y'], top_k=int(q['signals']), extra_steps=len(x_val))
            for c in senoidales.columns:
                x_val[c] = (senoidales[c]).values[len(x_train):]

            for c in senoidales.columns:
                x_train[c] = (senoidales[c]).values[:len(x_train)]   

            # Ordenar por fecha por seguridad
            x_train = x_train.sort_values('ds')
            # Aplicar promedio progresivo mes a mes
            x_train['Promedio_Historico'] = (
                x_train
                .groupby('month')['y']
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
                .astype(int)
            )
            # Aplicar promedio progresivo mes a mes
            x_train['Std_Historico'] = (
                x_train
                .groupby('month')['y']
                .expanding()
                .std()
                .reset_index(level=0, drop=True)
            )

            #for l in ['12', '24', '36', '48']:
            #    x_train[f'relacion_y_prom_hist_{l}'] = (x_train[f'y_last_{l}'] / (x_train['Promedio_Historico'] + 1e-6)).replace(0, np.nan)
            #    x_train[f'relacion_Std_Historico_{l}'] = (x_train[f'y_last_{l}'] / (x_train['Std_Historico'] + 1e-6)).replace(0, np.nan)
            
            #for l in ['12', '24', '36', '48']:
            #    x_val[f'relacion_y_prom_hist_{l}'] = (x_val[f'y_last_{l}'] / (x_val['Promedio_Historico'] + 1e-6)).replace(0, np.nan)
            #    x_val[f'relacion_Std_Historico_{l}'] = (x_val[f'y_last_{l}'] / (x_val['Std_Historico'] + 1e-6)).replace(0, np.nan)

            x_val['upper'] = x_val['Promedio_Historico']+x_val['Std_Historico']
            x_val['lower'] = x_val['Promedio_Historico']-x_val['Std_Historico']

            x_train['upper'] = x_train['Promedio_Historico']+x_train['Std_Historico']
            x_train['lower'] = x_train['Promedio_Historico']-x_train['Std_Historico']

            '''features+= ['upper', 'lower', 
                        #'Promedio_Historico', 
                        'Std_Historico',# 'signal'
                        ]'''
            
            #features+= list(senoidales.columns)
            
            #features+= ['y_last_12', 'y_last_24', 'y_last_36']
            '''features += ['y_last_3', 'y_last_4','y_last_5', 'y_last_6', 'y_last_7', 
                        'y_last_8', 'y_last_9', 'y_last_10', 'y_last_11', 'y_last_12',
                        'y_last_13', 'y_last_14', 'y_last_15', 'y_last_16', 'y_last_17',
                        'y_last_18', 'y_last_19', 'y_last_20', 'y_last_21', 'y_last_22',
                        'y_last_23', 'y_last_24', 'y_last_25', 'y_last_26', 'y_last_27',
                        'y_last_28', 'y_last_29', 'y_last_30', 'y_last_31', 'y_last_32',
                        'y_last_33', 'y_last_34', 'y_last_35', 'y_last_36']'''

            
            correlaciones_con_y = x_val.select_dtypes(include='number').corr()['target'].sort_values(ascending=False).reset_index()
            important_ones = correlaciones_con_y[ abs(correlaciones_con_y['target'])>.4 ]
            #features = list(important_ones['index'])[1:]

            # Matrices de XGBoost
            dtrain = xgb.DMatrix(x_train[features], label=x_train['y'], feature_names=features)
            dval = xgb.DMatrix(x_val[features], label=x_val['target'], feature_names=features)

            # Matriz con toda la historia
            param = {
                'max_depth': int(q['max_depth']),
                'colsample_bytree': q['colsample_bytree'],
                'subsample': q['subsample'],
                'seed': 0,
                'verbosity': 0,
                'alpha': q['alpha'],
                'eta': q['eta'],
                'lambda': q['lambdaa'],
                'tree_method': 'hist',
                'eval_metric': 'mape'
            }

            # Train XGBoost model on training set
            xgb_model_train = xgb.train(
                param,
                dtrain,
                num_boost_round=int(q['num_boost_round']),
                early_stopping_rounds=150,
                verbose_eval=False,
                evals=[(dval, 'val')])
            print('Predicton')
            # PREDICCION
            cutoff_date = pd.to_datetime(cutoff_date)
            dff = pd.DataFrame(pd.date_range(cutoff_date,
                                            cutoff_date + relativedelta(months=12),
                                            freq='ME'),
                                            columns=['ds'])
            dff['ds'] = dff['ds'].apply(lambda x: x.replace(day=1))
            dff = utilities.features_from_date(dff, ds='ds')
            
            historicos = (
                pd.concat([x_train, x_val])
                .groupby('month')['y']
                .agg(Promedio_Historico='mean', Std_Historico='std')
                .reset_index())

            dff = dff.merge(historicos)
            dff = utilities.lags_future_features(dff=dff, 
                                                inference_data= x_train, 
                                                n_lags_start=1, 
                                                n_lags_end=48, 
                                                ds='ds')
            print(dff)

            
            senoidales, _, _ = utilities.generar_senoidales_exogenas(pd.concat([x_train, x_val])['y'], 
                                                                    top_k=int(q['signals']), 
                                                                    extra_steps=len(dff))

            for c in senoidales.columns:
                dff[c] = (senoidales[c]).values[len(pd.concat([x_train, x_val])):]

            print('Check')

            dff['upper'] = dff['Promedio_Historico']+dff['Std_Historico']
            dff['lower'] = dff['Promedio_Historico']-dff['Std_Historico']
            dff_matrix = xgb.DMatrix(dff[features], feature_names=features)
            dff['yhat'] = xgb_model_train.predict(dff_matrix)

            dff = dff[[self.ds, 'yhat']].copy()
            dff['dsm'] = dff[self.ds].apply(lambda x: int(str(x).replace('-', '')[: 6]))
            #dff['week_number'] = dff['ds'].dt.strftime('%U').astype(int)
            dff['yhat'] = dff['yhat'].round().astype(int)
            dff['yhat'] = dff['yhat'].apply(lambda x: 0 if x < 0 else x)
            #dff = dff.merge(mean_std, on='week_number', how='left')

            #dff['Upper'] = dff['Upper'].interpolate(method='linear')
            #dff['Lower'] = dff['Lower'].interpolate(method='linear')
            #dff['mean'] = dff['mean'].interpolate(method='linear')

            #dff['yhat'] = np.where(dff['yhat'] > dff['Upper'], dff['Upper'],
            #                       np.where(dff['yhat'] < dff['Lower'], dff['Lower'], dff['yhat']))
            #dff = dff[[self.ds, 'yhat', 'Upper', 'Lower', 'mean']].copy()

            dff[self.id] = category
            dff['forecast_date'] = cutoff_date
            return dff
            dff.to_csv(self.file_forecast_2, mode='a')

        except Exception as e:
            print('Data is too short to even train')
            print(f'Data lenght {len(data)}')
            print(e)


    def obj_xgb_do_month(self, eta=None,
                               max_depth=None,
                               subsample=None,
                               colsample_bytree=None,
                               alpha=None,
                               lambdaa=None,
                               years=None,
                               months=None,
                               num_boost_round=None,
                               signals=None,
                               data=None
                               ):
        """
            Funcion Objetivo / Objetive Function
        """
        try:
            ds = 'ds'
            features = ['quarter', 'month', 'year', 'dayofyear', 'weekofyear']
            """
                Funcion Objetivo / Objetive Function
            """
            data['dsm'] = data[ds].apply(lambda x: int(str(x).replace('-', '')[: 6]))

            #data = utilities.features_from_date(data, ds=ds)
            x_train, x_val = utilities.split_data_val(data=data,
                                                    train_years=int(years),
                                                    months_val=int(months),
                                                    date=ds)
            
            x_train = utilities.features_from_date(x_train, 'ds')            
            
            historicos = (
                x_train
                .groupby('month')['y']
                .agg(Promedio_Historico='mean', Std_Historico='std')
                .reset_index())
            
            x_train = utilities.laggin_features(2, 36, x_train, 'ds')

            x_val = utilities.lags_future_features(dff=x_val.rename(columns={'y':'target'}), 
                                                    inference_data=x_train,
                                                    n_lags_start= 2,
                                                    n_lags_end= 36,
                                                    ds='ds')

            x_val = utilities.features_from_date(x_val, 'ds')
            #print('Hola3')
            
            x_val = x_val.merge(historicos, on=['month'], how='inner')
            #print('Senoidales')

            # Senoidales
            senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train['y'], top_k=int(signals), extra_steps=len(x_val))
            #print(senoidales)
            #print('Paso antes')
            for c in senoidales.columns:
                x_val[c] = (senoidales[c]).values[len(x_train):]

            #print('Es aca?')
            for c in senoidales.columns:
                x_train[c] = (senoidales[c]).values[:len(x_train)]
            #print('Hola4')
            

            # Ordenar por fecha por seguridad
            x_train = x_train.sort_values('ds')
            # Aplicar promedio progresivo mes a mes
            x_train['Promedio_Historico'] = (
                x_train
                .groupby('month')['y']
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
                .astype(int)
            )
            # Aplicar promedio progresivo mes a mes
            x_train['Std_Historico'] = (
                x_train
                .groupby('month')['y']
                .expanding()
                .std()
                .reset_index(level=0, drop=True)
            )

            #print('Hola5')
            #for l in ['12', '24', '36', '48']:
            #    x_train[f'relacion_y_prom_hist_{l}'] = (x_train[f'y_last_{l}'] / (x_train['Promedio_Historico'] + 1e-6)).replace(0, np.nan)
            #    x_train[f'relacion_Std_Historico_{l}'] = (x_train[f'y_last_{l}'] / (x_train['Std_Historico'] + 1e-6)).replace(0, np.nan)
            
            #for l in ['12', '24', '36', '48']:
            #    x_val[f'relacion_y_prom_hist_{l}'] = (x_val[f'y_last_{l}'] / (x_val['Promedio_Historico'] + 1e-6)).replace(0, np.nan)
            #    x_val[f'relacion_Std_Historico_{l}'] = (x_val[f'y_last_{l}'] / (x_val['Std_Historico'] + 1e-6)).replace(0, np.nan)

            x_val['upper'] = x_val['Promedio_Historico']+x_val['Std_Historico']
            x_val['lower'] = x_val['Promedio_Historico']-x_val['Std_Historico']

            x_train['upper'] = x_train['Promedio_Historico']+x_train['Std_Historico']
            x_train['lower'] = x_train['Promedio_Historico']-x_train['Std_Historico']
            
            '''features+= ['upper', 'lower', 
                        #'Promedio_Historico', 
                        'Std_Historico',# 'signal'
                        ]'''
            
            #features+= list(senoidales.columns)
            
            #features+= ['y_last_12', 'y_last_24', 'y_last_36']

            '''features += ['y_last_3', 'y_last_4','y_last_5', 'y_last_6', 'y_last_7', 
                        'y_last_8', 'y_last_9', 'y_last_10', 'y_last_11', 'y_last_12',
                        'y_last_13', 'y_last_14', 'y_last_15', 'y_last_16', 'y_last_17',
                        'y_last_18', 'y_last_19', 'y_last_20', 'y_last_21', 'y_last_22',
                        'y_last_23', 'y_last_24', 'y_last_25', 'y_last_26', 'y_last_27',
                        'y_last_28', 'y_last_29', 'y_last_30', 'y_last_31', 'y_last_32',
                        'y_last_33', 'y_last_34', 'y_last_35', 'y_last_36']'''

            correlaciones_con_y = x_val.select_dtypes(include='number').corr()['target'].sort_values(ascending=False).reset_index()
            important_ones = correlaciones_con_y[ abs(correlaciones_con_y['target'])>.4 ]
            #features = list(important_ones['index'])[1:]
            
            # Matrices de XGBoost
            dtrain = xgb.DMatrix(x_train[features], label=x_train['y'], feature_names=features)
            dval = xgb.DMatrix(x_val[features], label=x_val['target'], feature_names=features)
        
            param = {
                'max_depth': int(max_depth),
                'colsample_bytree': colsample_bytree,
                'subsample': subsample,
                'seed': 0,
                'verbosity': 0,
                'alpha': alpha,
                'eta': eta,
                'lambda': lambdaa,
                'tree_method': 'hist',
                'eval_metric': 'mape'
            }
            # Train XGBoost model on training set
            xgb_model_train = xgb.train(
                param,
                dtrain,
                num_boost_round=int(num_boost_round),
                early_stopping_rounds=150,
                verbose_eval=False,
                evals=[(dval, 'val')])
    
            # Evaluate model on test set and return score
            x_val['yhat'] = xgb_model_train.predict(dval)
            x_val['yhat'] = x_val['yhat'].astype(int)

            #y_acc = x_val[['dsm', 'y']].groupby('dsm', as_index=False).sum().rename(columns={'y': 'y_acc'})
            #x_val = x_val.merge(y_acc)
            #x_val['y_weighted'] = x_val['y'] / x_val['y_acc']

            x_val['error'] = x_val.apply(utilities.mape, args=('yhat', 'target'), axis=1)

            #x_val['w_error'] = x_val['error'] * x_val['y_weighted']
            #x_val = x_val[['dsm', 'yhat', 'y', 'y_weighted', 'w_error']].groupby('dsm').sum()
            #x_val['acc'] = 100 - x_val['w_error']
            x_val['acc'] = 100 - x_val['error']
            return round(x_val['acc'].mean(), 2)
    
        except Exception as e:
            print(e)
            return 0
        
    def fit_xgb_do_month(self, data=None, category=None, cutoff_date=None):

        # group data
        #data[self.ds] = pd.to_datetime(data[self.ds])
        #data = data.set_index(self.ds).groupby([self.id, 'familia']).resample('ME')[self.y].sum().reset_index()
        #data['dsm'] = data[self.ds].apply(lambda x: int(str(x).replace('-', '')[: 6]))
        #data = data[data[self.id].isin([category])]
        data['ds'] = pd.to_datetime(data['ds'])
        data = data[data['ds']<cutoff_date]
        data['ds'] = data['ds'].apply(lambda x: x.replace(day=1))

        if data.shape[0] > 24:

            # train model
            pbounds = {
                'eta': (.1, .6),
                'max_depth': (3, 15),
                'colsample_bytree': (.3, .8),
                'subsample': (.4, .85),
                'lambdaa': (1, 10),
                'months': (3, 4),
                'years': (4, 8),
                'alpha': (4, 10),
                'num_boost_round': (25, 100),
                'signals':(1, 1)
            }

            # try:
            xgb_partial = partial(self.obj_xgb_do_month, data=data)

            # Initialize Bayesian optimizer
            optimizer = BayesianOptimization(f=xgb_partial,
                                             pbounds=pbounds,
                                             random_state=42,
                                             verbose=1,
                                             allow_duplicate_points=False)

            # Perform Bayesian optimization
            optimizer.maximize(init_points=10, n_iter=30)

            dic_params = dict()
            dic_params[self.id] = category
            dic_params = optimizer.max['params']
            dic_params['accuracy'] = optimizer.max['target']

            return dic_params
            #self.to_json(self.hyper_params, category=f'xgb_dft_{self.label}_{category}',
            #             dic={f'xgb_dft_{self.label}_{category}': dic_params})

            # except IndexError:
            #    print('index error')
            # except KeyError:
            #    print('problem with data detected')
        else:
            print(f"{category}: {data.shape[0]}")
