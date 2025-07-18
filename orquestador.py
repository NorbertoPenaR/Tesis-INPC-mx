# orquestador
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import fit
import predict
print('Hello')

class orchestrator:

    def __init__(self,
                data=None,
                fecha_d_corte=None,
                iteraciones=None,
                frequencia=None,
                horizonte=None,
                modelo=None,
                metrica=None
                ):
        
        self.data = data
        self.fecha_d_corte = fecha_d_corte
        self.iteraciones = iteraciones
        self.frequencia = frequencia
        self.horizonte = horizonte
        self.modelo = modelo
        self.metrica = metrica
        self.resultados_gen = []
        self.predicciones_gen = []
        
    def train_n_predict(self):
        if self.modelo=='lstm':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]

                parametros, _ = fit.fit_lstm(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                resultados, predicciones = predict.predict_lstm(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')
        
        elif self.modelo=='rnn':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                parametros, _ = fit.fit_rnn(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                
                resultados, predicciones = predict.predict_rnn(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')
        
        elif self.modelo=='deepAr':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                parametros, _ = fit.fit_deep_ar(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                
                resultados, predicciones = predict.predict_lstm(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')
        
        elif self.modelo=='transformer':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                parametros, _ = fit.fit_transformer(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                
                resultados, predicciones = predict.predict_transformer(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')

        elif self.modelo=='nhits':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]
                parametros, _ = fit.fit_transformer(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                
                resultados, predicciones = predict.predict_transformer(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')

        elif self.modelo=='xgb':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]

                parametros, _ = fit.fit_xgb(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                
                resultados, predicciones = predict.predict_xgb( 
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')

        elif self.modelo=='holt-winters':
            for id in self.data.unique_id.unique():
                print(f"Procesando ID: {id}")
                subset = self.data[self.data['unique_id'] == id]

                parametros, _ = fit.fit_holt_winters(
                    data=subset,
                    cutoff_date=self.fecha_d_corte,
                    iteraciones=self.iteraciones,
                    freak=self.frequencia,
                    horizon=self.horizonte,
                    Metric= self.metrica
                )
                
                resultados, predicciones = predict.predict_holt_winters(
                    config=parametros,
                    data=subset,
                    cutoff_date=self.fecha_d_corte
                )
                
                self.resultados_gen.append(resultados)
                self.predicciones_gen.append(predicciones)
            
            pd.concat(self.resultados_gen).to_csv(f'resultados-{self.modelo}-{self.fecha_d_corte}.csv')
            pd.concat(self.predicciones_gen).to_csv(f'forecast-{self.modelo}-{self.fecha_d_corte}.csv')

        elif self.modelo=='DVAE':
            resultados_gen = []
            predicciones_gen = []
            