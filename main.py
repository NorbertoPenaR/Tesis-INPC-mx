# main.py
# Copyright (c) 2024 Norberto P. R. – All rights reserved.
# Licensed for private use only.

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from orquestador import orchestrator
import process_data
from utiles import utilities
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Si no se cuenta con datos actualizados
print("🔗 Por favor descarga el archivo manualmente desde:")
print("https://www.inegi.org.mx/app/tabulados/default.aspx?nc=ca56_2018a&idrt=137&opc=t")
print("📁 Luego, súbelo aquí para continuar con el análisis.")

# Datos - Volumen de Venta en México - MABE.
pais = 'mex'

inpc_path_Q = 'ca56_2018a.csv'
inpc_Q = process_data.limpiar_csv_inegi(inpc_path_Q)
weekly_inpc = process_data.inpc_data_weekly(datos=inpc_Q)

#weekly_inpc.to_csv('inpc_historico_weekly.csv')
inpc_path_M = 'ca55_2018a.csv'
inpc_M = process_data.limpiar_csv_inegi(inpc_path_M)
monthly_inpc = process_data.inpc_monthly(datos=inpc_M)

print('Fechas - Simulación')
print('Historia disponible del indice nacional de precios al consumidor')
print('INPC semanal.')
print(weekly_inpc.ds.min())
print(weekly_inpc.ds.max())

# Metricas disponibles
# 'MAE' 'MAPE' 'RMSE' 'MSE'
# MAE is preferred when we dont want to penalize so much the outliers,
#  but in this case, it might be good; while MSE actually does. 

all_ids = ['Inflacion', 'Subyacente', 'Mercancias', 'Alimentos_bebidas_tabaco',
 'Mercancias_no_alimenticias', 'Servicios', 'Vivienda', 'Educacion_colegiaturas',
 'Otros servicios', 'No_subyacente', 'Agropecuarios', 'Frutas_verduras',
 'Pecuarios', 'Energeticos_tarifas_autorizadas_por_el_gobierno', 'Energeticos',
 'Tarifas_autorizadas_por_el_gobierno']
# Modelos disponibles
# lstm - Done
# rnn - Done
# deepAr - To test
# transformer - To test
# nhits - Not going to
# xgb - Done
# holt-winters - Done
# D^3VAE - Needs testing. 

# Download hourly data for the last 5 days

# Para DeepAr el horizonte tiene q ser 26 para obtener buenos resultados. 52 needs to be tested again. 
# H = 26 / 6 meses hacia el futuro. 
# Y el learning rate necesita seguir fijo, 25 iteraciones. 

simulation_dates_mex = utilities.ultimos_dias_meses(n=4, frecuencia=1, referencia='2025-02-01')
#print(simulation_dates_mex)
#aires_ids = sellin_weekly[sellin_weekly['familia']=='AIAC'].unique_id.unique()
#print(len(aires_ids))

print(simulation_dates_mex)
'''transformaciones_map = {
    'diff': 0,
    'diff_logp1': 1,
    'pct':2,
    'logp1':3,
    'none':4,
    'diff2':5
}'''
clima_data = pd.read_csv('historico_clima_mex.csv')
clima_data['unique_id'] = clima_data['Estado']+'_'+clima_data['tipo']
clima_data.rename(columns={'fecha':'ds', 'valor':'y'}, inplace=True)
clima_data['ds'] = pd.to_datetime(clima_data['ds'])
#clima_data = clima_data.set_index('ds').sort_index().resample('Me').sum().reset_index()

#ids_clima = ['Nacional_PREC', 'Nacional_TMAX', 'Nacional_TMIN', 'Nacional_TMED']
transformaciones = ['diff', 'diff_logp1', 'pct', 'logp1', 'diff2']# 'none',

all_models = ['xgb', 'holt_winters', 'lstm', 'rnn', 'deepAr', 'transformer']
all_models = ['lstm', 'rnn', 'xgb', 'deepAr']

# 4 out of 7, falta deepar y Transformer, también integrar DVAE de alguna manera. 
all_models =['transformer'
            #'deepAr',
            #'rnn'
            #'fft', 
            #'holt_winters'
            #'xgb',
            #'holt_winters'
            ]

#all_models = ['lstm', #'rnn',# 'xgb', 'holt_winters'
             #]

             # RNN ya lo logro, no hay que moverle más. 
             # Ahora es replicarlo con lstm. Analizar si disminuye o no. 
             # En principio siguen el mismo camino"" " solo que lstm tiene
             # ventaja dada su arquitectura, y sus celulas ocultas. -Neuroas-
             # Puertas para olvidar o recordar. 

simulation_dates = utilities.ultimos_dias_meses(n=6, frecuencia=3, referencia='2025-01-01')
#for señales in [2, 4, 6, 8, 10, 12, 14]:
for j in range(3):
            
    for sm_date in tqdm(simulation_dates,  desc=f"Running Simulation"):
        #j = 1
        #print(sm_date)
        for math_model in all_models:
        #for trns in transformaciones:
            orc  =  orchestrator(data = monthly_inpc, #monthly_inpc, #weekly_inpc,
                                fecha_d_corte= sm_date, 
                                iteraciones= 15,
                                frequencia='W-mon',
                                horizonte=26, # 52/2 = 26
                                modelo=math_model,
                                metrica='MAE',
                                ids=['Inflacion'],#ids_clima,#all_ids,#['Pecuarios'],
                                mes_val=4,
                                features=j,
                                transformacion='diff',
                                signals=4)
            
            orc.train_n_predict()

'''total = 
cut_date = '2025-01-01'
busqueda_hyper = 10
resultados = []
for skt in sellin_file.sku.unique():
    try:
        sellin_weekly_sku = sellin_weekly[sellin_weekly['sku'] == skt].rename(columns={'sku':'unique_id'})
        
        if len(sellin_weekly_sku[sellin_weekly_sku['ds']<=cut_date])>=80:
            parameters, acc_val = fit.fit_nhits(sellin_weekly_sku[['ds', 'unique_id', 'y']], 
                                                cut_date, 
                                                10)
            
            preds = predict.predict_nhits(parameters, 
                                        sellin_weekly_sku[['ds', 'unique_id', 'y']], 
                                        cut_date)
            
            resultados.append(preds)
        else:
            print('Data is too short to train')
            print(sellin_weekly_sku[sellin_weekly_sku['ds']<=cut_date])
    except Exception as e:
        print('There is not enough data to train the model')
        print(len(sellin_weekly_sku[sellin_weekly_sku['ds']<=cut_date]))
resultados = pd.concat(resultados)
resultados.to_csv(f'nhits_test_{pais}.csv')'''


'''
import yfinance as yf
# Define the ticker symbol
ticker = "LULU"
intel_data = yf.download(ticker, 
                         interval='1d', 
                         period='5d',
                         start='2010-01-02'
                         ).reset_index()
intel_data.rename(columns={'Close':'y', 'Date':'ds'}, inplace=True)
#intel_data['unique_id'] = ticker
intel_data['ds'] = pd.to_datetime(intel_data['ds']).dt.tz_localize(None)
df = pd.DataFrame()
df['y'] = intel_data['y']
df['ds'] = intel_data['ds']
df['unique_id'] = ticker
print(df)'''