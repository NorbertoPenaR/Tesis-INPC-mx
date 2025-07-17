# main.py
import pandas as pd
import numpy as np
import fit
import predict

pais = 'do'
sellin_file = pd.read_csv(f'sellin_files_05_30/sellin_{pais}_2025_05_30.csv')
sellin_file['ds'] = pd.to_datetime(sellin_file['ds'])
sellin_weekly = sellin_file.set_index('ds').groupby(['sku',
                    'familia', 'CLASS_ABC']).resample('W-mon')['y'].sum().reset_index()

total = 
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
resultados.to_csv(f'nhits_test_{pais}.csv')