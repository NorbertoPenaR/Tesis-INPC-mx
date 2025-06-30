# Processing Data
import pandas as pd
import numpy as np
from utiles import utilities
def inpc_data_weekly(total_info=None):
    inpc = pd.read_csv('indice_nacional_precios_consumidor.csv')
    # Aplicarlo al dataframe
    inpc['ds'] = inpc['ds'].apply(utilities.convertir_fecha_quincenal)
    inpc_inflacion = utilities.calcular_inflacion_por_columnas(inpc).dropna()
    inpc_inflacion = inpc_inflacion.set_index('ds').sort_index()
    inpc_weekly = inpc_inflacion.resample('W-mon').sum().reset_index()#.interpolate(method='linear')
    inpc_weekly = inpc_weekly.replace(0, np.nan)
    inpc_weekly = inpc_weekly.interpolate(method='linear')
    columnas = ['Inflacion', 'Subyacente', 'Mercancias',
                'Alimentos_bebidas_tabaco', 'Mercancias_no_alimenticias', 'Servicios',
                'Vivienda', 'Educacion_colegiaturas', 'Otros servicios',
                'No_subyacente', 'Agropecuarios', 'Frutas_verduras', 'Pecuarios',
                'Energeticos_tarifas_autorizadas_por_el_gobierno', ' Energeticos',
                'Tarifas_autorizadas_por_el_gobierno']
    venta = 4#spans[3]
    for col in columnas:
        col_clean = col.strip()  # limpiar espacios en blanco si hay
        inpc_weekly[f'{col_clean}_EMA'] = inpc_weekly[col].ewm(span=venta, adjust=False).mean()
        inpc_weekly[f'{col_clean}_MA'] = inpc_weekly[col].rolling(window=venta, min_periods=1).mean()

    #return inpc_weekly.melt(id_vars=['ds'], var_name='unique_id', value_name='y')
    # Se regresa por partes
    # Que regrese todo el conjunto de datos mejor.
    if total_info ==1:
        inpc_total_info = inpc_weekly.melt(id_vars=['ds'], 
                                           var_name='unique_id', 
                                           value_name='y')
        return inpc_total_info
    elif total_info == 0:

        inpc_weekly_00_20 = inpc_weekly[ (inpc_weekly['ds']<'2020-01-01') &
                                         (inpc_weekly['ds']>'2000-01-01') ]
        inpc_weekly_20_25 = inpc_weekly[ (inpc_weekly['ds']>='2020-01-01') ]
        inpc_long_00_20 = inpc_weekly_00_20.melt(id_vars=['ds'], 
                                                var_name='unique_id', 
                                                value_name='y')
        inpc_long_20_25 = inpc_weekly_20_25.melt(id_vars=['ds'], 
                                                var_name='unique_id', 
                                                value_name='y')
        return inpc_long_00_20, inpc_long_20_25


def inpc_data_quincenal():
    inpc = pd.read_csv('indice_nacional_precios_consumidor.csv')
    # Aplicarlo al dataframe
    inpc['ds'] = inpc['ds'].apply(utilities.convertir_fecha_quincenal)
    inpc_inflacion = utilities.calcular_inflacion_por_columnas(inpc).dropna()
    inpc_inflacion = inpc_inflacion.set_index('ds').sort_index()
    inpc_inflacion.reset_index(inplace=True)

    columnas = ['Inflacion', 'Subyacente', 'Mercancias',
                'Alimentos_bebidas_tabaco', 'Mercancias_no_alimenticias', 'Servicios',
                'Vivienda', 'Educacion_colegiaturas', 'Otros servicios',
                'No_subyacente', 'Agropecuarios', 'Frutas_verduras', 'Pecuarios',
                'Energeticos_tarifas_autorizadas_por_el_gobierno', ' Energeticos',
                'Tarifas_autorizadas_por_el_gobierno']

    inpc_weekly_00_20 = inpc_inflacion[ (inpc_inflacion['ds']<'2020-01-01') & (inpc_inflacion['ds']>'2000-01-01') ]
    inpc_weekly_20_25 = inpc_inflacion[ (inpc_inflacion['ds']>='2020-01-01') ]
    inpc_long_00_20 = inpc_weekly_00_20.melt(id_vars=['ds'], var_name='unique_id', value_name='y')
    inpc_long_20_25 = inpc_weekly_20_25.melt(id_vars=['ds'], var_name='unique_id', value_name='y')
    return inpc_long_00_20, inpc_long_20_25

# Temperatura Semanal
def temp_data():
    temp_mex = pd.read_csv('historico_clima_mex.csv').rename(columns={'fecha':'ds'})#.dropna()
    temp_mex['ds'] = pd.to_datetime(temp_mex['ds'])
    temp_weekly = temp_mex.set_index('ds').groupby(['tipo', 'Estado']).resample('W-mon')['valor'].mean().reset_index()
    #temp_weekly = temp_weekly['valor'].replace(0, np.nan)
    #temp_weekly.dropna(subset='valor', inplace=True)
    temp_weekly['valor'] = temp_weekly['valor'].interpolate(method='polynomial', order=2)