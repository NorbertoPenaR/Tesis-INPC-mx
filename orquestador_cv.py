# orquestador_cv.py
# orquestador.py
# Copyright (c) 2024 Norberto P. R. – All rights reserved.
# Licensed for private use only.

from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
#import fit
#import predict
import matplotlib.pyplot as plt 
import fit_bayes_opt as fit
import predict_bayes as predict
import predict_bayes_cv
import cross_v# as fit
from tqdm import tqdm
import os
from datetime import datetime
import time

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error


# Formato: YYYYMMDD-HHMM
timestamp = datetime.now().strftime("%Y%m%d-%H%M")

class orchestrator_cv:

    def __init__(self,
                data=None,
                fecha_d_corte=None,
                iteraciones=None,
                frequencia=None,
                horizonte=None,
                modelo=None,
                metrica=None,
                ids=None,
                mes_val=None,
                features=None,
                transformacion=None,
                signals=None
                ):
        
        self.data = data
        self.fecha_d_corte = fecha_d_corte
        self.iteraciones = iteraciones
        self.frequencia = frequencia
        self.horizonte = horizonte
        self.modelo = modelo
        self.metrica = metrica
        self.features = features
        self.resultados_gen = []
        self.predicciones_gen = []
        self.performance_gen = []
        self.signals=signals
        self.transformation = transformacion
        self.ids = ids
        self.mes_val = mes_val
        self.str_date = str(self.fecha_d_corte)[: 10]

        # Se crean las carpetas si no existen
        os.makedirs("resultados", exist_ok=True)
        os.makedirs("pronosticos", exist_ok=True)
        os.makedirs("performance", exist_ok=True)

        self.timestamp = timestamp
        self.file_resultados = (
            f"resultados/resultados-"
            f"{self.modelo}-{self.features}-{self.transformation}-"
            f"{self.str_date}-{self.mes_val}-{self.timestamp}-"
            f"sgls-{self.signals}.csv"
        )

        self.file_forecast = (
            f"pronosticos/forecast-"
            f"{self.modelo}-{self.features}-{self.transformation}-"
            f"{self.str_date}-{self.mes_val}-{self.timestamp}-"
            f"sgnls-{self.signals}.csv"
        )

        self.file_performance = (
            f"performance/performance-"
            f"{self.modelo}-{self.features}-{self.transformation}-"
            f"{self.str_date}-{self.mes_val}-{self.timestamp}-"
            f"sgnls-{self.signals}.csv"
        )
        
        #self.file_resultados = f'resultados/resultados-{self.modelo}-{self.str_date}-{self.mes_val}.csv'
        #self.file_forecast = f'pronosticos/forecast-{self.modelo}-{self.str_date}-{self.mes_val}.csv'
        
    def train_n_predict(self):

        print('Modelo')
        print(self.modelo)

        if self.modelo=='avg_naive':
            for id in tqdm(self.ids, desc=f"Entrenando {self.modelo}"):
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]

                start_time = time.time()

                parametros, _ = cross_v.fit_avg_rwd_naive_cv(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    feats=self.features,
                    transf=self.transformation,
                    signals=self.signals
                )

                predicciones, resultados = predict_bayes_cv.predict_avg_naive_cv(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )

                end_time = time.time()
                elapsed_time = end_time - start_time

                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time

                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)

        elif self.modelo=='fft':
            for id in tqdm(self.ids, desc=f"Entrenando {self.modelo}"):
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]

                start_time = time.time()
                parametros, _ = fit.fit_fft(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    #feats=self.features,
                    transf=self.transformation,
                    signals=self.signals
                )

                _, predicciones, resultados = predict.predict_dft(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )

                end_time = time.time()
                elapsed_time = end_time - start_time

                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time
                resultados['years']= parametros['years']
                resultados['months']= parametros['months']

                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)

        elif self.modelo=='lstm':
            for id in tqdm(self.ids, desc=f"Entrenando {self.modelo}"):
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                try:
                    start_time = time.time()
                    parametros, _ = cross_v.fit_lstm_cv(
                        data=subset,
                        cutoff_date=self.fecha_d_corte,
                        iteraciones=self.iteraciones,
                        freak=self.frequencia,
                        horizon=self.horizonte,
                        Metric=self.metrica,
                        Mes_val=self.mes_val,
                        feats=self.features,
                        transf=self.transformation,
                        signals=self.signals
                    )

                    performance, predicciones, resultados = predict_bayes_cv.predict_lstm_cv(
                        config=parametros,
                        data=subset,
                        cutoff_date=self.fecha_d_corte
                    )


                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    resultados['fecha_d_corte'] = self.str_date
                    resultados['execution_time'] = elapsed_time
                    resultados['years']= parametros['years']
                    resultados['months']= parametros['months']
                    resultados['input_size']= parametros['input_size']
                    resultados['neurons']= parametros['neurons']
                    resultados['layers']= parametros['layers']
                    resultados['max_steps'] = parametros['max_steps']
                    #resultados['learning_rate'] = parametros['learning_rate']
                    resultados['signals'] = parametros['signals']
                    
                    self.performance_gen.append(performance)
                    self.resultados_gen.append(resultados)
                    self.predicciones_gen.append(predicciones)
                except Exception as e:
                    #print(e)
                    print('Data is probably to short to even train'
                    ' and get good predictions')
                    print(subset.shape)
            
            pd.concat(self.performance_gen).to_csv(self.file_performance)
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)
        
        elif self.modelo=='rnn':
            for id in self.ids:
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                start_time = time.time()
                try:

                    parametros, _ = cross_v.fit_rnn_cv(
                        data=subset,
                        cutoff_date=self.fecha_d_corte,
                        iteraciones=self.iteraciones,
                        freak=self.frequencia,
                        horizon=self.horizonte,
                        Metric=self.metrica,
                        Mes_val=self.mes_val,
                        feats=self.features,
                        transf=self.transformation,
                        signals=self.signals
                    )
                    
                    performance, predicciones, resultados = predict_bayes_cv.predict_rnn_cv(
                        config=parametros,
                        data=subset,
                        cutoff_date=self.fecha_d_corte
                    )
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    resultados['fecha_d_corte'] = self.str_date
                    resultados['execution_time'] = elapsed_time
                    resultados['years']= parametros['years']
                    resultados['months']= parametros['months']
                    resultados['input_size']= parametros['input_size']
                    resultados['neurons']= parametros['neurons']
                    resultados['layers']= parametros['layers']
                    resultados['max_steps']= parametros['max_steps']
                    resultados['signals'] = parametros['signals']

                    self.performance_gen.append(performance)
                    self.resultados_gen.append(resultados)
                    self.predicciones_gen.append(predicciones)
                except Exception as e:
                    #print(e)
                    print('Data is probably to short to even train'
                    ' and get good predictions')
                    print(subset.shape)

            pd.concat(self.performance_gen).to_csv(self.file_performance)
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)
        
        elif self.modelo=='deepAr':
            print('Entro a DeepAr')
            for id in self.ids:
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                start_time = time.time()
                parametros, _ = cross_v.fit_deep_ar_cv(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    feats=self.features,
                    transf=self.transformation,
                    signals=self.signals
                )
                
                performance, predicciones, resultados = predict_bayes_cv.predict_deepAr_cv(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time
                resultados['years']= parametros['years']
                resultados['months']= parametros['months']
                resultados['input_size']= parametros['input_size']
                resultados['neurons']= parametros['neurons']
                resultados['layers']= parametros['layers']
                resultados['max_steps']= parametros['max_steps']
                resultados['trajectories']= parametros['trajectories']
                resultados['learning_rate']= parametros['learning_rate']
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)
        
        elif self.modelo=='transformer':
            for id in self.ids:
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                print('Subset')
                print(subset)
                
                start_time = time.time()
                parametros, _ = cross_v.fit_transformer_cv(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    feats=self.features,
                    transf=self.transformation,
                    signals=self.signals
                )
                
                performance, predicciones, resultados = predict_bayes_cv.predict_transformer_cv(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time
                resultados['input_size'] = parametros['input_size']
                resultados['neurons'] = parametros['neurons']
                resultados['conv_size'] = parametros['conv_size']
                resultados['n_heads'] = parametros['n_heads']
                resultados['max_steps'] = parametros['max_steps']
                resultados['learning_rate'] = parametros['learning_rate']

                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)

        elif self.modelo=='nhits':
            for id in self.ids:
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                start_time = time.time()
                parametros, _ = fit.fit_transformer(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    feats=self.features
                )
                
                resultados, predicciones = predict.predict_transformer(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)

        elif self.modelo=='xgb':
            for id in tqdm(self.ids, desc=f"Entrenando {self.modelo}"):
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                print('subset lenght')
                print(len(subset))
                print(subset.ds.max())
                print(subset.ds.min())

                #offset = 10  # or any small constant appropriate for your data
                #subset['y'] = subset['y'] + offset
                if len(subset) >= 52:

                    start_time = time.time()
                    parametros, _ = cross_v.fit_xgb_cv(
                        data=subset,
                        cutoff_date=self.fecha_d_corte,
                        iteraciones=self.iteraciones,
                        freak=self.frequencia,
                        horizon=self.horizonte,
                        Metric=self.metrica,
                        Mes_val=self.mes_val,
                        feats=self.features,
                        transf=self.transformation,
                        signals=self.signals
                    )
                    
                    #try:
                    performance, predicciones, resultados = predict_bayes_cv.predict_xgb_cv( 
                        config=parametros,
                        data=subset,
                        cutoff_date=self.fecha_d_corte
                    )

                    #resultados['xgb_og'] = resultados['xgb_og'] - offset
                    #resultados['xgb_og'] = resultados['xgb_og'].clip(lower=0)
                    #resultados['y'] = resultados['y'] - offset
                    resultados['max_depth']= parametros['max_depth']
                    resultados['colsample_bytree']= parametros['colsample_bytree']
                    resultados['subsample']= parametros['subsample']
                    resultados['alpha']= parametros['alpha']
                    resultados['eta']= parametros['eta']
                    resultados['lambdaa']= parametros['lambdaa']
                    resultados['num_boost_round'] = parametros['num_boost_round'] 
                    resultados['years']= parametros['years']
                    resultados['months']= parametros['months']
                    resultados['signals'] = parametros['signals']

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    resultados['fecha_d_corte'] = self.str_date
                    resultados['execution_time'] = elapsed_time
                    self.resultados_gen.append(resultados)
                    self.predicciones_gen.append(predicciones)
                    #except Exception as e:
                    #    print(e)
                    #    print(f'No params for {id}')
            
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)
        
        # Does not need Exo
        elif self.modelo=='holt_winters':
            for id in tqdm(self.ids, desc=f"Entrenando {self.modelo}"):
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                start_time = time.time()
                parametros, _ = cross_v.fit_holt_winters_cv(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    transf=self.transformation,
                    #signals=self.signals
                )
                
                resultados, predicciones = predict_bayes_cv.predict_holt_winters_cv(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time
                resultados['years']= parametros['years']
                resultados['months']= parametros['months']
                
                resultados['trend_type'] = parametros['trend_type']
                resultados['seasonal_type'] = parametros['seasonal_type']
                resultados['damped_trend'] = parametros['damped_trend']
                resultados['use_boxcox'] = parametros['use_boxcox']
                resultados['seasonal_periods'] = parametros['seasonal_periods']
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)

            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)
            
        elif self.modelo=='d3vae':
            for id in self.ids:
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                print('Subset')
                print(subset)
                
                start_time = time.time()
                parametros, _ = cross_v.fit_dvae_cv(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric=self.metrica,
                    Mes_val=self.mes_val,
                    feats=self.features,
                    transf=self.transformation,
                    signals=self.signals
                )
                
                performance, predicciones, resultados = predict_bayes_cv.predict_d3vae_cv(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                resultados['fecha_d_corte'] = self.str_date
                resultados['execution_time'] = elapsed_time
                resultados['input_size'] = parametros['input_size']
                resultados['layers'] = parametros['layers']
                resultados['neurons'] = parametros['neurons']
                resultados['dimension'] = parametros['dimension']
                resultados['beta_kl'] = parametros['beta_kl']
                resultados['max_steps'] = parametros['max_steps']
                resultados['teacher_forcing'] = parametros['teacher_forcing']

                self.performance_gen.append(performance)
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)

            pd.concat(self.performance_gen).to_csv(self.file_performance)
            pd.concat(self.resultados_gen).to_csv(self.file_resultados)
            pd.concat(self.predicciones_gen).to_csv(self.file_forecast)            

            