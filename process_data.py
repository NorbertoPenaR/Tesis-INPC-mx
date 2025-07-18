# Processing Data
import pandas as pd
import numpy as np
from utiles import utilities

def inpc_data_weekly(total_info=1, datos = None):
    '''
    Parameters
    total_info: 1 returns the whole history
                0 returns two dataframes, the first ones contains the information 
                from 2000 till 2019-12-31, the other one returns from '2020-01-01' foward. ¿onward?
    '''
    #inpc = pd.read_csv('indice_nacional_precios_consumidor.csv')
    inpc = datos.copy()
    # Aplicarlo al dataframe
    inpc['ds'] = inpc['ds'].apply(utilities.convertir_fecha_quincenal)
    inpc_inflacion = utilities.calcular_inflacion_por_columnas(inpc).dropna()
    inpc_inflacion = inpc_inflacion.set_index('ds').sort_index()
    inpc_weekly = inpc_inflacion.resample('W-mon').sum().reset_index()
    inpc_weekly = inpc_weekly.replace(0, np.nan)
    inpc_weekly = inpc_weekly.interpolate(method='linear')
    
    columnas = ['Inflacion', 'Subyacente', 'Mercancias', 'Alimentos_bebidas_tabaco',
        'Mercancias_no_alimenticias', 'Servicios', 'Vivienda', 'Educacion_colegiaturas',
        'Otros servicios', 'No_subyacente', 'Agropecuarios', 'Frutas_verduras',
        'Pecuarios', 'Energeticos_tarifas_autorizadas_por_el_gobierno', 'Energeticos',
        'Tarifas_autorizadas_por_el_gobierno']
    
    inpc_weekly.columns = [col.strip() for col in ['ds', 
        'Inflacion', 'Subyacente', 'Mercancias', 'Alimentos_bebidas_tabaco',
        'Mercancias_no_alimenticias', 'Servicios', 'Vivienda', 'Educacion_colegiaturas',
        'Otros servicios', 'No_subyacente', 'Agropecuarios', 'Frutas_verduras',
        'Pecuarios', 'Energeticos_tarifas_autorizadas_por_el_gobierno', 'Energeticos',
        'Tarifas_autorizadas_por_el_gobierno'
    ]]

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

def limpiar_csv_inegi(path):
    # 1. Leer los nombres reales (fila 2, índice 2)
    encabezados_largos = pd.read_csv(
        path,
        skiprows=2,
        nrows=1,
        encoding='latin1',
        quotechar='"',
        header=None
    ).values.flatten().tolist()

    # 2. Extraer solo la parte útil (lo último después de la última coma)
    column_names = ['ds'] + [
        str(col).strip().split(',')[-1].strip() if pd.notna(col) else f'col_{i+1}'
        for i, col in enumerate(encabezados_largos[1:])
    ]

    # 3. Leer los datos reales desde la fila 10 (índice 9)
    df = pd.read_csv(
        path,
        skiprows=9,
        encoding='latin1',
        quotechar='"',
        header=None,
        names=column_names
    )

    # 4. Eliminar filas basura (Tipo, Fuente, Atención, etc.)
    excluir = ['Tipo', 'Unidad', 'Cifra', 'Fecha', 'Fuente', 'Atención', 'consulta']
    df = df[~df['ds'].astype(str).str.contains('|'.join(excluir), na=False)]

    # 5. Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')

    # 6. Convertir columnas numéricas (excepto 'ds')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.reset_index(drop=True).dropna()


    

