# main.py
import pandas as pd
import numpy as np
import fit
import predict
from orquestador import orchestrator
import process_data
# Si no se cuenta con datos actualizados
print("🔗 Por favor descarga el archivo manualmente desde:")
print("https://www.inegi.org.mx/app/tabulados/default.aspx?nc=ca56_2018a&idrt=137&opc=t")
print("📁 Luego, súbelo aquí para continuar con el análisis.")

# Datos - Volumen de Venta en México - MABE.
pais = 'do'
sellin_file = pd.read_csv(f'sellin_files_05_30/sellin_{pais}_2025_05_30.csv')
sellin_file['ds'] = pd.to_datetime(sellin_file['ds'])
sellin_weekly = sellin_file.set_index('ds').groupby(['sku',
                    'familia', 'CLASS_ABC']).resample('W-mon')['y'].sum().reset_index()

inpc_path = 'ca56_2018a.csv'
inpc = process_data.limpiar_csv_inegi(inpc_path)
weekly_inpc = process_data.inpc_data_weekly(datos=inpc)

print(inpc.ds.min())
print(inpc.ds.max())

print(weekly_inpc.ds.min())
print(weekly_inpc.ds.max())

pronostico = orchestrator(data = weekly_inpc,
                        fecha_d_corte= '2022-06-01', 
                        iteraciones= 5,
                        frequencia='W-mon',
                        horizonte=15,
                        modelo='rnn')


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