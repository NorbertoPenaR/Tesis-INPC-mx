# main_cv.py
# Copyright (c) 2025 Norberto P. R. – All rights reserved.
# Licensed for private use only.

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from orquestador_cv import orchestrator_cv
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

inpc_path_Q = 'ca56_2018a-2025_10_14.csv'
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
pais = 'mex'
sellin_file = pd.read_csv(f'sellin_files/sellin_{pais}-20251001_1632.csv')
sellin_file['ds'] = pd.to_datetime(sellin_file['ds'])
sellin_sku = sellin_file.set_index('ds').groupby(['sku', 'CLASS_ABC', 'familia']).resample('ME')['y'].sum().reset_index()
sellin_fams = sellin_file.set_index('ds').groupby(['familia']).resample('ME')['y'].sum().reset_index()
sellin_fams.rename(columns={'familia':'unique_id'}, inplace=True)

all_ids = ['Inflacion', 'Subyacente', 'Mercancias', 'Alimentos_bebidas_tabaco',
 'Mercancias_no_alimenticias', 'Servicios', 'Vivienda', 'Educacion_colegiaturas',
 'Otros servicios', 'No_subyacente', 'Agropecuarios', 'Frutas_verduras',
 'Pecuarios', 'Energeticos_tarifas_autorizadas_por_el_gobierno', 'Energeticos',
 'Tarifas_autorizadas_por_el_gobierno']

clima_mex = pd.read_csv('estados_principales_tmax_aires.csv')

print(list(clima_mex.unique_id.unique()))
all_models =[
            #'transformer',
            #'deepAr',
            #'rnn',
            #'fft', 
            #'holt_winters'
            #'xgb',
            #'holt_winters'
            'lstm'
            #'d3vae'
            #'avg_naive'
            ]

simulation_dates = utilities.ultimos_dias_meses(n=6, frecuencia=3, referencia='2025-01-01')
#for señales in [2, 4, 6, 8, 10, 12, 14]:
for j in range(0,1):
    #print(j)
    #j = j+1
    for sm_date in tqdm(simulation_dates,  desc=f"Running Simulation"):
        
        print(sm_date)
        for math_model in all_models:
        #for trns in transformaciones:
            orc  =  orchestrator_cv(data = sellin_fams,#clima_mex, #monthly_inpc, #weekly_inpc,
                            fecha_d_corte= sm_date, 
                            iteraciones= 20,
                            frequencia='ME',
                            horizonte=12, # 52/2 = 26 # Si la frequencia es mensual, entonces
                            # tienes que ajustar bien.
                            modelo=math_model,
                            metrica='MAE',
                            ids= list(sellin_fams['unique_id'].unique()),
                            #ids=list(clima_mex.unique_id.unique()),
                            #ids=['Inflacion'],#ids_clima,#all_ids,#['Pecuarios'],
                            mes_val=4,
                            features=j,
                            transformacion='diff',
                            signals=8
                            )
            
            orc.train_n_predict()