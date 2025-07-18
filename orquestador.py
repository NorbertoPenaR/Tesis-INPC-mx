# orquestador
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import fit
import predict

class orchestrator(fit, predict):

    def __init__(self,
                data=None,
                fecha_d_corte=None,
                iteraciones=None,
                frequencia=None,
                horizonte=None,
                modelo=None,
                ):
        
        if modelo=='lstm':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_lstm(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_lstm(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')
        
        elif modelo=='rnn':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_lstm(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_lstm(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')
        
        elif modelo=='deepAr':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_lstm(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_lstm(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')
        
        elif modelo=='transformer':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_transformer(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_transformer(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')

        elif modelo=='nhits':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_transformer(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_transformer(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')

        elif modelo=='xgb':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_xgb(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_xgb(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')

        elif modelo=='holt-winters':
            resultados_gen = []
            predicciones_gen = []
            for id in data.unique_id.unique():
                parametros, _ = fit.fit_holt_winters(data=data, 
                                            cutoff_date=fecha_d_corte, 
                                            iteraciones=iteraciones, 
                                            freak=frequencia,
                                            horizon=horizonte)
                
                resultados, predicciones = predict.predict_holt_winters(config= parametros, 
                                                                data=data, 
                                                                cutoff_date=fecha_d_corte)
                
                resultados_gen.append(resultados)
                predicciones_gen.append(predicciones)
            
            pd.concat(resultados_gen).to_csv(f'resultados-{modelo}-{fecha_d_corte}.csv')
            pd.concat(predicciones_gen).to_csv(f'forecast-{modelo}-{fecha_d_corte}.csv')

        elif modelo=='DVAE'
            resultados_gen = []
            predicciones_gen = []