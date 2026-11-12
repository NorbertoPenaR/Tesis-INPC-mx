# best_model_dist.py
# Copyright (c) 2024 Norberto P. R. – All rights reserved.
# Licensed for private use only.

import pandas as pd
import numpy as np
from utiles import utilities
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from functools import partial
from bayes_opt import BayesianOptimization
import xgboost as xgb
import warnings
import accuracy
warnings.filterwarnings("ignore")

class xgb_senoidales:
    ds = 'ds'
    id = 'sku'
    
    def predict_xgb_month(self, data=None, category=None, dic_params=None, cutoff_date=None):
        try:
            q = dic_params
            data['ds'] = pd.to_datetime(data['ds'])
            data = data[data['ds']<cutoff_date]
            data['ds'] = data['ds'].apply(lambda x: x.replace(day=1))
            
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
            senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train['y'], top_k=int(q['signals']), extra_steps=12 + len(x_val))
            # Entrenamiento
            for c in senoidales.columns:
                x_train[c] = (senoidales[c]).values[:len(x_train)]
            # Validación
            for c in senoidales.columns:
                x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))] 

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
            
            features = ['quarter', 'month', 'year', 'dayofyear', 'weekofyear']
            #features += list(senoidales.columns)[:-1]
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
            for c in senoidales.columns:
                dff[c] = (senoidales[c]).values[len(pd.concat([x_train, x_val])):]
            dff_matrix = xgb.DMatrix(dff[features], feature_names=features)

            dff['upper'] = dff['Promedio_Historico']+dff['Std_Historico']
            dff['lower'] = dff['Promedio_Historico']-dff['Std_Historico']

            dff['yhat'] = xgb_model_train.predict(dff_matrix)
            dff['yhat'] = dff['yhat']*q['reduction']

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
                               reduction=None,
                               data=None
                               ):
        """
            Funcion Objetivo / Objetive Function
        """
        try:
            ds = 'ds'
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
            x_val = x_val.merge(historicos, on=['month'], how='inner')
            
            # Senoidales
            senoidales, _, _ = utilities.generar_senoidales_exogenas(x_train['y'], top_k=int(signals), extra_steps= len(x_val))
            # Entrenamiento
            for c in senoidales.columns:
                x_train[c] = (senoidales[c]).values[:len(x_train)]
            # Validación
            for c in senoidales.columns:
                x_val[c] = (senoidales[c]).values[len(x_train):(len(x_train) + len(x_val))]
            
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
            
            
            #features+= ['y_last_12', 'y_last_24', 'y_last_36']

            '''features += ['y_last_3', 'y_last_4','y_last_5', 'y_last_6', 'y_last_7', 
                        'y_last_8', 'y_last_9', 'y_last_10', 'y_last_11', 'y_last_12',
                        'y_last_13', 'y_last_14', 'y_last_15', 'y_last_16', 'y_last_17',
                        'y_last_18', 'y_last_19', 'y_last_20', 'y_last_21', 'y_last_22',
                        'y_last_23', 'y_last_24', 'y_last_25', 'y_last_26', 'y_last_27',
                        'y_last_28', 'y_last_29', 'y_last_30', 'y_last_31', 'y_last_32',
                        'y_last_33', 'y_last_34', 'y_last_35', 'y_last_36']'''

            #correlaciones_con_y = x_val.select_dtypes(include='number').corr()['target'].sort_values(ascending=False).reset_index()
            #important_ones = correlaciones_con_y[ abs(correlaciones_con_y['target'])>.4 ]
            #features = list(important_ones['index'])[1:]

            features = ['quarter', 'month', 'year', 'dayofyear', 'weekofyear']
            #features += list(senoidales.columns)[:-1]
            
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
            x_val['yhat'] = x_val['yhat']*reduction

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
        #print('Datos Pre Entrenamiento')
        #print(data)

        if data.shape[0] > 24:

            # train model
            pbounds = {
                'eta': (.1, .6),
                'max_depth': (3, 20),
                'colsample_bytree': (.3, .8),
                'subsample': (.4, .85),
                'lambdaa': (1, 10),
                'months': (2, 4),
                'years': (4, 8),
                'alpha': (4, 10),
                'num_boost_round': (25, 100),
                'signals':(1, 5),
                'reduction':(.4, 1) # .3 on lower
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
            optimizer.maximize(init_points=15, n_iter=40) # 15 ,  40

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


#for m, ymm in zip(['02', '03', '04', '05'], [202503, 202504, 202505, 202506]):
import accuracy
# To be forecasted
#  do, pe, ar,
# DONE
# gt, cr, pa, hn, ni, sv
def simulacion(country='mex',
               date_file='2025_07_24',
               month_list=['01','02', '03', '04', '05', '06'], 
               year_months= [202502,202503, 202504, 202505, 202506, 202507],
               clases=['A', 'B', 'C'],
               i_d_A=None,
               i_d_B=None,
               alpha_v3=.2
               ):
    
    '''
    Params:
    alpha_v2
    alpha_v3 
    Returns:
    resultados_fam, resultado_gen, whole_predictions, sku_metrics
    '''
    whole_predictions = []
    resultados_fam = []
    resultado_gen = []
    sku_metrics = []
    pais = country
    sellin_file = pd.read_csv(f'sellin_files/sellin_{pais}_{date_file}.csv')
    sellin_file['ds'] = pd.to_datetime(sellin_file['ds'])
    if country in ['do', 'ni', 'pa']:
            sellin_file = sellin_file[~sellin_file['familia'].isin(['AIAC', 'GLOB'])]
            
    sellin_sku = sellin_file.set_index('ds').groupby(['sku', 'CLASS_ABC', 'familia']).resample('ME')['y'].sum().reset_index()
    sellin_fams = sellin_file.set_index('ds').groupby(['familia']).resample('ME')['y'].sum().reset_index()
    sellin_abc = sellin_file.set_index('ds').groupby(['CLASS_ABC']).resample('ME')['y'].sum().reset_index()
    sellin_total = sellin_file.set_index('ds').resample('ME')['y'].sum().reset_index()

    #print('Skus Totales en el archivo')
    #print(len(sellin_sku.sku.unique()))

    for m, ymm in zip(month_list, year_months) :
    #for m, ymm in zip(['04'], [202505]):

        sellin_fams.rename(columns={'familia':'unique_id'}, inplace=True)
        sellin_abc.rename(columns={'CLASS_ABC':'unique_id'}, inplace=True)
        results_rnn = []
        for abc in clases:

            clase_abc = sellin_abc[sellin_abc['unique_id']==abc]
            #print('Head of Date')
            #print(clase_abc.head())
            #print('skus en la clase')
            #print(len(clase_abc.sku.unique()))
            # Los datos son menores o iguales
            # Entonces, lo que hará será tomar info antes de esa fecha. 
            # Este ejercicio es para Febrero
            H = xgb_senoidales()
            # Ajuste 
            p = H.fit_xgb_do_month(clase_abc, abc,  cutoff_date=f'2025-{m}-01')
            # Prediccion
            preds = H.predict_xgb_month(clase_abc, abc, p, cutoff_date=f'2025-{m}-01')
            preds.rename(columns={'sku':'unique_id'}, inplace=True)
            #params, acc_val = fit.fit_xgb_do_month(clase_abc, f'2025-{m}-01', 10, 'ME', 12, 'MAPE')
            #results, Y_hat_df = predict.predict_xgb_month(params, clase_abc, f'2025-{m}-01')
            results_rnn.append(preds)

            sellin_sku_A = sellin_sku[sellin_sku['CLASS_ABC']==abc]

            # Se toman los últimos años dado el mes.
            # Osea, la fecha de corte. 
            sellin_sku_A_cut = sellin_sku_A[ (sellin_sku_A['ds']<f'2025-{m}-01') 
                                            & (sellin_sku_A['ds']>=f'2023-{m}-01')
                                            ]
            
            # Puede ser Adaptado para que tome una distribución más. 
            cutted_picture = sellin_sku_A[ (sellin_sku_A['ds']<f'2025-{m}-01')]

            suma_total = sellin_sku_A_cut.groupby(['ds'])['y'].sum().reset_index(name='acc')
            pesos = sellin_sku_A_cut.merge(suma_total, on='ds', how='inner')
            pesos['weight'] = (pesos['y'])/pesos['acc']
            pesos['month']=pesos.ds.dt.month
            pesos['year']=pesos.ds.dt.year

            # Pesos Recientes
            cutoff_dt = f'2025-{m}-01'
            desired = pd.to_datetime(cutoff_dt)
            desface = desired - pd.DateOffset(months=3)
            sellin_sku_A_recent_cut = sellin_sku_A[sellin_sku_A['ds'] > desface ]
            recent_contribution = sellin_sku_A_recent_cut.groupby(['sku'])['y'].sum().reset_index()
            recent_total_sale = recent_contribution['y'].sum()
            recent_contribution['total'] = recent_total_sale
            recent_contribution['weight_recent'] = recent_contribution['y']/recent_contribution['total']

            # Peso - por Familia en cada Mes. 
            # Se Agrupa por mes y familia, para ver la contribucion de cada familia a lo largo de los años
            # Posteriores a la fecha de corte. 
            cutted_picture# = sellin_sku_A_cut.copy()
            cutted_picture['month'] = cutted_picture.ds.dt.month
            fam_contribution = cutted_picture.groupby(['month', 'familia'])['y'].sum().reset_index()
            # Acumulado mensual. 
            acc_mensual = cutted_picture.groupby(['month'])['y'].sum().reset_index().rename(columns={'y':'acc'})
            # Se hace el merge. 
            fam_percentage = fam_contribution.merge(acc_mensual)
            fam_percentage['weight_fam'] = ((fam_percentage['y'])/fam_percentage['acc'])+1

            # Paso 1: Calcular el promedio mensual por SKU con familia
            contribucion_promediada = pesos.groupby(['month', 'sku', 'familia'])['weight'].mean().reset_index()
            #print('SKUS totales')
            #print(len(contribucion_promediada.sku.unique()))
            # Agrega la columna de ajuste basado en la familia
            #contribucion_promediada['ajuste'] = contribucion_promediada['familia'].map(ajuste_por_familia).fillna(1)
            contribucion_promediada = contribucion_promediada.merge(fam_percentage)
            contribucion_promediada['peso_ajustado'] = contribucion_promediada['weight']*contribucion_promediada['weight_fam']
            
            # Aplica el ajuste
            # Renormaliza dentro de cada mes
            contribucion_promediada['weight_normalized'] = (
                contribucion_promediada
                .groupby('month')['peso_ajustado']
                .transform(lambda x: x / x.sum())
            )

            # Ajuste Reciente. 
            combined_weights = contribucion_promediada.merge(
                recent_contribution[['sku', 'weight_recent']],
                on='sku',
                how='left'
            )
            combined_weights['weight_recent'] = combined_weights['weight_recent'].fillna(0)
            alpha = alpha_v3  # 70% peso histórico, 30% peso reciente
            combined_weights['peso_final'] = (
                alpha * combined_weights['weight_normalized'] +
                (1 - alpha) * combined_weights['weight_recent']
            )
            combined_weights['peso_final'] = (
                combined_weights
                .groupby('month')['peso_final']
                .transform(lambda x: x / x.sum())
            )

            preds_A = preds
            preds_A['month']=preds_A.ds.dt.month
            reparto = preds_A.merge(
                combined_weights[['sku', 'month', 'peso_final']],
                on=['month'],
                how='inner'
            )
            reparto.rename(columns={'yhat':'total'}, inplace=True)
            reparto['yhat'] = reparto['total'] * reparto['peso_final']
            reparto['yhat'] = reparto['yhat']*i_d# increase_decrese = 
            a,b,c,d,e = accuracy.metricas_analytics_periodo(sellin_file, 
                                                            reparto[['ds', 'sku', 'yhat']], 
                                                            ymm, 
                                                            'ME')

            resultados_fam.append(a)
            resultado_gen.append(b)
            whole_predictions.append(reparto)
            sku_metrics.append(d)
    
    resultados_fam = pd.concat(resultados_fam)        
    resultado_gen = pd.concat(resultado_gen)        
    whole_predictions = pd.concat(whole_predictions)        
    sku_metrics = pd.concat(sku_metrics) 

    return resultados_fam, resultado_gen, whole_predictions, sku_metrics

def forecast(country='mex',
            date_file='2025_07_24',
            month_list=['08'], 
            year_months= [202509],
            clases=['A', 'B', 'C'],
            i_d=None,
            alpha_v3=.2
            ):
    
    whole_predictions = []
    pais = country
    sellin_file = pd.read_csv(f'sellin_files/sellin_{pais}_{date_file}.csv')
    sellin_file['ds'] = pd.to_datetime(sellin_file['ds'])
    sellin_sku = sellin_file.set_index('ds').groupby(['sku', 'CLASS_ABC', 'familia']).resample('ME')['y'].sum().reset_index()
    sellin_fams = sellin_file.set_index('ds').groupby(['familia']).resample('ME')['y'].sum().reset_index()
    sellin_abc = sellin_file.set_index('ds').groupby(['CLASS_ABC']).resample('ME')['y'].sum().reset_index()

    print('SKUS')
    print(len(sellin_sku.sku.unique()))

    for m, ymm in zip(month_list, year_months) :
    #for m, ymm in zip(['04'], [202505]):

        sellin_fams.rename(columns={'familia':'unique_id'}, inplace=True)
        sellin_abc.rename(columns={'CLASS_ABC':'unique_id'}, inplace=True)
        results_rnn = []

        for abc in clases:
            clase_abc = sellin_abc[sellin_abc['unique_id']==abc]
            print('Head of Date')
            print(clase_abc.head())
            #print('skus en la clase')
            #print(len(clase_abc.sku.unique()))
            # Los datos son menores o iguales
            # Entonces, lo que hará será tomar info antes de esa fecha. 
            # Este ejercicio es para Febrero
            H = xgb_senoidales()
            # Ajuste 
            p = H.fit_xgb_do_month(clase_abc, abc,  cutoff_date=f'2025-{m}-15')
            # Prediccion
            preds = H.predict_xgb_month(clase_abc, abc, p, cutoff_date=f'2025-{m}-15')
            preds.rename(columns={'sku':'unique_id'}, inplace=True)
            #params, acc_val = fit.fit_xgb_do_month(clase_abc, f'2025-{m}-01', 10, 'ME', 12, 'MAPE')
            #results, Y_hat_df = predict.predict_xgb_month(params, clase_abc, f'2025-{m}-01')
            results_rnn.append(preds)

            sellin_sku_A = sellin_sku[sellin_sku['CLASS_ABC']==abc]
            sellin_sku_A_cut = sellin_sku_A[ (sellin_sku_A['ds']<f'2025-{m}-15')
                                            & (sellin_sku_A['ds']>=f'2023-{m}-15')
                                            ]
            
            # Puede ser Adaptado para que tome una distribución más. 
            cutted_picture = sellin_sku_A[ (sellin_sku_A['ds']<f'2025-{m}-15')]

            suma_total = sellin_sku_A_cut.groupby(['ds'])['y'].sum().reset_index(name='acc')
            pesos = sellin_sku_A_cut.merge(suma_total, on='ds', how='inner')
            pesos['weight'] = (pesos['y'])/pesos['acc']
            pesos['month']=pesos.ds.dt.month
            pesos['year']=pesos.ds.dt.year

            # Pesos Recientes
            cutoff_dt = f'2025-{m}-15'
            desired = pd.to_datetime(cutoff_dt)
            desface = desired - pd.DateOffset(months=3)
            sellin_sku_A_recent_cut = sellin_sku_A[sellin_sku_A['ds'] > desface ]
            recent_contribution = sellin_sku_A_recent_cut.groupby(['sku'])['y'].sum().reset_index()
            recent_total_sale = recent_contribution['y'].sum()
            recent_contribution['total'] = recent_total_sale
            recent_contribution['weight_recent'] = recent_contribution['y']/recent_contribution['total']

            # Peso - por Familia en cada Mes. 
            # Se Agrupa por mes y familia, para ver la contribucion de cada familia a lo largo de los años
            # Posteriores a la fecha de corte. 
            cutted_picture# = sellin_sku_A_cut.copy()
            cutted_picture['month'] = cutted_picture.ds.dt.month
            fam_contribution = cutted_picture.groupby(['month', 'familia'])['y'].sum().reset_index()
            # Acumulado mensual. 
            acc_mensual = cutted_picture.groupby(['month'])['y'].sum().reset_index().rename(columns={'y':'acc'})
            # Se hace el merge. 
            fam_percentage = fam_contribution.merge(acc_mensual)
            fam_percentage['weight_fam'] = ((fam_percentage['y'])/fam_percentage['acc'])+1

            # Paso 1: Calcular el promedio mensual por SKU con familia
            contribucion_promediada = pesos.groupby(['month', 'sku', 'familia'])['weight'].mean().reset_index()
            contribucion_promediada = contribucion_promediada.merge(fam_percentage)
            contribucion_promediada['peso_ajustado'] = contribucion_promediada['weight']*contribucion_promediada['weight_fam']
            
            contribucion_promediada['weight_normalized'] = (
                contribucion_promediada
                .groupby('month')['peso_ajustado']
                .transform(lambda x: x / x.sum())
            )

            # Ajuste Reciente. 
            combined_weights = contribucion_promediada.merge(
                recent_contribution[['sku', 'weight_recent']],
                on='sku',
                how='left'
            )
            
            combined_weights['weight_recent'] = combined_weights['weight_recent'].fillna(0)
            alpha = alpha_v3  # 70% peso histórico, 30% peso reciente
            combined_weights['peso_final'] = (
                alpha * combined_weights['weight_normalized'] +
                (1 - alpha) * combined_weights['weight_recent']
            )

            combined_weights['peso_final'] = (
                combined_weights
                .groupby('month')['peso_final']
                .transform(lambda x: x / x.sum())
            )

            preds_A = preds
            preds_A['month']=preds_A.ds.dt.month
            reparto = preds_A.merge(
                combined_weights[['sku', 'month', 'peso_final']],
                on=['month'],
                how='inner'
            )
            reparto.rename(columns={'yhat':'total'}, inplace=True)
            reparto['yhat'] = reparto['total']*reparto['peso_final']
            reparto['yhat'] = reparto['yhat']*i_d# increase_decrese = 

            whole_predictions.append(reparto)

    whole_predictions = pd.concat(whole_predictions)
    return whole_predictions
     
#pd.concat(whole_predictions).to_csv(f'xgb-monthly-for-division-{pais}-2025-{m}-01.csv')