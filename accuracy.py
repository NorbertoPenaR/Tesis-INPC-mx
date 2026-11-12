import pandas as pd
import numpy as np

def mape(data=None, yhat=None, y=None):
    diff = abs(data[yhat] - data[y]) / (data[y] + 1e-8)
    if diff > 1:
        return 100
    elif diff <= 0:
        return 0
    else:
        return diff * 100


def sape(data=None, yhat=None, y=None):
    diff = abs(data[yhat] - data[y]) / (data[yhat] + 1e-8)
    if diff > 1:
        return 100
    elif diff <= 0:
        return 0
    else:
        return diff * 100

def counting_acc_ranges(data_frame=None, columns_to_analyse=None):
    s70 = []
    s50 = []
    s30 = []
    s10 = []
    s0 = []
    for c in columns_to_analyse:
        s70.append(data_frame[ data_frame[c]>=70 ].shape[0])
        s50.append(data_frame[ (data_frame[c]>=50 )  & (data_frame[c]<70)].shape[0])
        s30.append(data_frame[ (data_frame[c]>=30 )  & (data_frame[c]<50)].shape[0])
        s10.append(data_frame[ (data_frame[c]>0 )  & (data_frame[c]<30)].shape[0])
        s0.append(data_frame[ data_frame[c]==0 ].shape[0])
    
    A = pd.DataFrame(index=columns_to_analyse)
    A['above_70%'] = s70
    A['50-70%'] = s50
    A['30-50%'] = s30
    A['less_30%_above_0%'] = s10
    A['0%'] = s0
    return A
    
def metricas_analytics_periodo(cr, cr_sept, ym, periodo=None ):
    '''
    Returns 
    final_month_fam, final_month_abc, final_month_ym, serie_temporal, overall_performance_month
    '''
    # Periodo = 'ME' o 'W-mon'
    #Forecast
    cr_sept.dropna(inplace=True)        
    cr_sept['yhat'] = cr_sept['yhat'].astype(float).astype(int)
    cr_sept['ds'] = pd.to_datetime(cr_sept['ds'])
    #Data
    cr['ds'] = pd.to_datetime(cr['ds'])
    cr = cr.set_index('ds').groupby(['sku','familia','CLASS_ABC']).resample(periodo)['y'].sum().reset_index()
    cr['ds'] = cr['ds'].apply(lambda x: x.replace(day=15))
    cr_sept['ds'] = cr_sept['ds'].apply(lambda x: x.replace(day=15))

    #IBP / Estadistico CEAM
    #ibp_cr['yhat_ceam'] = ibp_cr['yhat_ceam'].str.replace(',', '').astype(int)
    
    cr_sept_merge = cr_sept.merge(cr,on=['ds','sku'],how='inner')       # Se une el Forecast con los Datos Reales.
    #print(cr_sept_merge)
    cr_sept_merge['yearmonth'] = cr_sept_merge['ds'].apply(lambda x: int(str(x).replace('-', '')[: 6])) # Se crea la variable Yearmonth
    cr_sept_by_yearmonth = cr_sept_merge.groupby(['yearmonth','sku','familia', 'CLASS_ABC'])[['yhat', 'y']].sum().reset_index()#.drop(columns=['Unnamed: 0']) # Se agrupa por Yearmonth

    cr_sept_by_yearmonth['err_aa_y'] = cr_sept_by_yearmonth.apply(mape , args=('yhat', 'y'), axis=1)               # Calculo de Mape y
    cr_sept_by_yearmonth['err_aa_yh'] = cr_sept_by_yearmonth.apply(sape , args=('yhat', 'y'), axis=1)              # Calculo de Mape yhat

    cr_sept_by_yearmonth['acc_sku_aa_y'] = round(100 - cr_sept_by_yearmonth['err_aa_y'], 4) ## Asertvidad por SKU - Mape Y
    cr_sept_by_yearmonth['acc_sku_aa_yhat'] = round(100 - cr_sept_by_yearmonth['err_aa_yh'], 4) ## Asertvidad por SKU - Mape Yhat

    serie_temporal = cr_sept_by_yearmonth[cr_sept_by_yearmonth.yearmonth == ym].copy() # Serie Temporal para Visual de PowerBi.
    cr_sept_by_yearmonth_fam = cr_sept_by_yearmonth[cr_sept_by_yearmonth.yearmonth == ym].copy()
    cr_sept_by_yearmonth_abc = cr_sept_by_yearmonth[cr_sept_by_yearmonth.yearmonth == ym].copy()

    # VENTA ACUMULADA POR CLASE - FAMLILY, ABC AND SKU.
    y_acc = cr_sept_by_yearmonth.groupby('yearmonth')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta Mensual
    yhat_acc = cr_sept_by_yearmonth.groupby('yearmonth')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts
    y_acc_fam = cr_sept_by_yearmonth_fam.groupby('familia')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta - Familia
    yhat_acc_fam = cr_sept_by_yearmonth_fam.groupby('familia')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts
    y_acc_abc = cr_sept_by_yearmonth_abc.groupby('CLASS_ABC')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta - ABC
    yhat_acc_abc = cr_sept_by_yearmonth_abc.groupby('CLASS_ABC')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts

    # Union de DataFrames con Sus Respectivos Acumulados, Familia, ABC y SKU.
    cr_sept_by_yearmonth_fam = cr_sept_by_yearmonth_fam.merge(y_acc_fam, on='familia',how='left') 
    cr_sept_by_yearmonth_fam = cr_sept_by_yearmonth_fam.merge(yhat_acc_fam, on='familia',how='left')
    cr_sept_by_yearmonth_abc = cr_sept_by_yearmonth_abc.merge(y_acc_abc,on='CLASS_ABC',how='left') 
    cr_sept_by_yearmonth_abc = cr_sept_by_yearmonth_abc.merge(yhat_acc_abc,on='CLASS_ABC',how='left')
    cr_sept_by_yearmonth = cr_sept_by_yearmonth.merge(y_acc,on='yearmonth',how='left') # Se unen los acumulados 
    cr_sept_by_yearmonth = cr_sept_by_yearmonth.merge(yhat_acc,on='yearmonth',how='left') 

    #Calculo de Weights in Family, ABC and SKU.
    cr_sept_by_yearmonth['y_w'] = cr_sept_by_yearmonth['y']/cr_sept_by_yearmonth['y_acc'] # SKU
    cr_sept_by_yearmonth['aa_w'] =  cr_sept_by_yearmonth['yhat']/cr_sept_by_yearmonth['aa'] # Weights based on the forecasted of analytics
    cr_sept_by_yearmonth_fam['y_w'] = cr_sept_by_yearmonth_fam['y']/cr_sept_by_yearmonth_fam['y_acc'] # FAM
    cr_sept_by_yearmonth_fam['aa_w'] = cr_sept_by_yearmonth_fam['yhat']/cr_sept_by_yearmonth_fam['aa']
    cr_sept_by_yearmonth_abc['y_w'] = cr_sept_by_yearmonth_abc['y']/cr_sept_by_yearmonth_abc['y_acc'] # ABC
    cr_sept_by_yearmonth_abc['aa_w'] = cr_sept_by_yearmonth_abc['yhat']/cr_sept_by_yearmonth_abc['aa']

    # Multiplicacion de Errores por los Weights de cada SKU de la Venta Total de cada CLASE. 
    cr_sept_by_yearmonth['e_aa_w_yh'] = cr_sept_by_yearmonth['err_aa_yh']*cr_sept_by_yearmonth['aa_w']      # YearMonth 
    cr_sept_by_yearmonth['e_aa_w'] = cr_sept_by_yearmonth['err_aa_y']*cr_sept_by_yearmonth['y_w']
    cr_sept_by_yearmonth_fam['e_aa_w_yh'] = cr_sept_by_yearmonth_fam['err_aa_yh']*cr_sept_by_yearmonth_fam['aa_w'] # FAMILY
    cr_sept_by_yearmonth_fam['e_aa_w'] = cr_sept_by_yearmonth_fam['err_aa_y']*cr_sept_by_yearmonth_fam['y_w']
    cr_sept_by_yearmonth_abc['e_aa_w_yh'] = cr_sept_by_yearmonth_abc['err_aa_yh']*cr_sept_by_yearmonth_abc['aa_w'] # ABC
    cr_sept_by_yearmonth_abc['e_aa_w'] = cr_sept_by_yearmonth_abc['err_aa_y']*cr_sept_by_yearmonth_abc['y_w']

    month_acc = cr_sept_by_yearmonth.groupby('yearmonth').sum().reset_index()[['yearmonth','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
    #month_acc['acc_ceam_h'] = 100 - month_acc['e_ceam_w_yh']
    month_acc['acc_aa_h'] = round(100 - month_acc['e_aa_w_yh'], 4)
    #month_acc['acc_ceam_y'] = 100 - month_acc['e_ceam_w']
    month_acc['acc_aa_y'] = round(100 - month_acc['e_aa_w'], 4)

    fam_month = cr_sept_by_yearmonth_fam.groupby('familia').sum().reset_index()[['familia','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
    #fam_month['acc_ceam_h'] = 100 - fam_month['e_ceam_w_yh']
    fam_month['acc_aa_h'] = round(100 - fam_month['e_aa_w_yh'], 4)
    #fam_month['acc_ceam_y'] = 100 - fam_month['e_ceam_w']
    fam_month['acc_aa_y'] = round(100 - fam_month['e_aa_w'], 4)
    fam_month['ym'] = ym
    
    abc_month = cr_sept_by_yearmonth_abc.groupby('CLASS_ABC').sum().reset_index()[['CLASS_ABC','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
    #abc_month['acc_ceam_h'] = 100 - abc_month['e_ceam_w_yh']
    abc_month['acc_aa_h'] = round(100 - abc_month['e_aa_w_yh'], 4)
    #abc_month['acc_ceam_y'] = 100 - abc_month['e_ceam_w']
    abc_month['acc_aa_y'] = round(100 - abc_month['e_aa_w'], 4)
    abc_month['ym'] = ym
    abc_month.rename(columns={'CLASS_ABC':'abc'},inplace=True)

    final_month_fam = fam_month[['familia','ym','y','yhat','acc_aa_h','acc_aa_y']]
    final_month_abc = abc_month[['abc','ym','y','yhat','acc_aa_h','acc_aa_y']]
    final_month_ym = month_acc[month_acc.yearmonth == ym][['yearmonth','y','yhat','acc_aa_h','acc_aa_y']]
    overall_performance_month = month_acc[['yearmonth','y','yhat','acc_aa_h','acc_aa_y']]

    return final_month_fam, final_month_abc, final_month_ym, serie_temporal, overall_performance_month


def metricas_analytics(cr, cr_sept, ym ):
    #Forecast
    cr_sept.dropna(inplace=True)        
    cr_sept['yhat'] = cr_sept['yhat'].astype(float).astype(int)
    cr_sept['ds'] = pd.to_datetime(cr_sept['ds'])
    #Data
    cr['ds'] = pd.to_datetime(cr['ds'])
    cr = cr.set_index('ds').groupby(['sku','familia','CLASS_ABC']).resample('W-Mon')['y'].sum().reset_index()
    #IBP / Estadistico CEAM
    #ibp_cr['yhat_ceam'] = ibp_cr['yhat_ceam'].str.replace(',', '').astype(int)

    cr_sept_merge = cr_sept.merge(cr,on=['ds','sku'],how='inner')       # Se une el Forecast con los Datos Reales.
    cr_sept_merge['yearmonth'] = cr_sept_merge['ds'].apply(lambda x: int(str(x).replace('-', '')[: 6])) # Se crea la variable Yearmonth
    #print(cr_sept_merge)
    cr_sept_by_yearmonth = cr_sept_merge.groupby(['yearmonth','sku','familia','CLASS_ABC'])[['yhat', 'y']].sum().reset_index()#.drop(columns=['Unnamed: 0']) # Se agrupa por Yearmonth
    #ibp_cr_aa = cr_sept_by_yearmonth.merge(ibp_cr, on=['sku','yearmonth'],how='inner')  #  Se juntan los dtos con IBP

    cr_sept_by_yearmonth['err_aa_y'] = cr_sept_by_yearmonth.apply(mape , args=('yhat', 'y'), axis=1)               # Calculo de Mape y
    cr_sept_by_yearmonth['err_aa_yh'] = cr_sept_by_yearmonth.apply(sape , args=('yhat', 'y'), axis=1)              # Calculo de Mape yhat

    cr_sept_by_yearmonth['acc_sku_aa_y'] = round(100 - cr_sept_by_yearmonth['err_aa_y'], 4) ## Asertvidad por SKU - Mape Y
    cr_sept_by_yearmonth['acc_sku_aa_yhat'] = round(100 - cr_sept_by_yearmonth['err_aa_yh'], 4) ## Asertvidad por SKU - Mape Yhat

    serie_temporal = cr_sept_by_yearmonth[cr_sept_by_yearmonth.yearmonth == ym].copy() # Serie Temporal para Visual de PowerBi.
    cr_sept_by_yearmonth_fam = cr_sept_by_yearmonth[cr_sept_by_yearmonth.yearmonth == ym].copy()
    cr_sept_by_yearmonth_abc = cr_sept_by_yearmonth[cr_sept_by_yearmonth.yearmonth == ym].copy()

    # VENTA ACUMULADA POR CLASE - FAMLILY, ABC AND SKU.
    y_acc = cr_sept_by_yearmonth.groupby('yearmonth')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta Mensual
    yhat_acc = cr_sept_by_yearmonth.groupby('yearmonth')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts
    y_acc_fam = cr_sept_by_yearmonth_fam.groupby('familia')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta - Familia
    yhat_acc_fam = cr_sept_by_yearmonth_fam.groupby('familia')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts
    y_acc_abc = cr_sept_by_yearmonth_abc.groupby('CLASS_ABC')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta - ABC
    yhat_acc_abc = cr_sept_by_yearmonth_abc.groupby('CLASS_ABC')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts

    # Union de DataFrames con Sus Respectivos Acumulados, Familia, ABC y SKU.
    cr_sept_by_yearmonth_fam = cr_sept_by_yearmonth_fam.merge(y_acc_fam, on='familia',how='left') 
    cr_sept_by_yearmonth_fam = cr_sept_by_yearmonth_fam.merge(yhat_acc_fam, on='familia',how='left')
    cr_sept_by_yearmonth_abc = cr_sept_by_yearmonth_abc.merge(y_acc_abc,on='CLASS_ABC',how='left') 
    cr_sept_by_yearmonth_abc = cr_sept_by_yearmonth_abc.merge(yhat_acc_abc,on='CLASS_ABC',how='left')
    cr_sept_by_yearmonth = cr_sept_by_yearmonth.merge(y_acc,on='yearmonth',how='left') # Se unen los acumulados 
    cr_sept_by_yearmonth = cr_sept_by_yearmonth.merge(yhat_acc,on='yearmonth',how='left') 

    #Calculo de Weights in Family, ABC and SKU.
    cr_sept_by_yearmonth['y_w'] = cr_sept_by_yearmonth['y']/cr_sept_by_yearmonth['y_acc'] # SKU
    #cr_sept_by_yearmonth['ceam_w'] =  cr_sept_by_yearmonth['yhat_ceam']/cr_sept_by_yearmonth['ceam'] # Peso en base al pronostico de ceam
    cr_sept_by_yearmonth['aa_w'] =  cr_sept_by_yearmonth['yhat']/cr_sept_by_yearmonth['aa'] # Weights based on the forecasted of analytics

    cr_sept_by_yearmonth_fam['y_w'] = cr_sept_by_yearmonth_fam['y']/cr_sept_by_yearmonth_fam['y_acc'] # FAM
    #cr_sept_by_yearmonth_fam['ceam_w'] = cr_sept_by_yearmonth_fam['yhat_ceam']/cr_sept_by_yearmonth_fam['ceam']
    cr_sept_by_yearmonth_fam['aa_w'] = cr_sept_by_yearmonth_fam['yhat']/cr_sept_by_yearmonth_fam['aa']

    cr_sept_by_yearmonth_abc['y_w'] = cr_sept_by_yearmonth_abc['y']/cr_sept_by_yearmonth_abc['y_acc'] # ABC
    #cr_sept_by_yearmonth_abc['ceam_w'] = cr_sept_by_yearmonth_abc['yhat_ceam']/cr_sept_by_yearmonth_abc['ceam']
    cr_sept_by_yearmonth_abc['aa_w'] = cr_sept_by_yearmonth_abc['yhat']/cr_sept_by_yearmonth_abc['aa']

    # Multiplicacion de Errores por los Weights de cada SKU de la Venta Total de cada CLASE. 
    cr_sept_by_yearmonth['e_aa_w_yh'] = cr_sept_by_yearmonth['err_aa_yh']*cr_sept_by_yearmonth['aa_w']      # YearMonth 
    #cr_sept_by_yearmonth['e_ceam_w_yh'] = cr_sept_by_yearmonth['err_ceam_yh']*cr_sept_by_yearmonth['ceam_w']
    cr_sept_by_yearmonth['e_aa_w'] = cr_sept_by_yearmonth['err_aa_y']*cr_sept_by_yearmonth['y_w']
    #cr_sept_by_yearmonth['e_ceam_w'] = cr_sept_by_yearmonth['err_ceam_y']*cr_sept_by_yearmonth['y_w']

    cr_sept_by_yearmonth_fam['e_aa_w_yh'] = cr_sept_by_yearmonth_fam['err_aa_yh']*cr_sept_by_yearmonth_fam['aa_w'] # FAMILY
    #cr_sept_by_yearmonth_fam['e_ceam_w_yh'] = cr_sept_by_yearmonth_fam['err_ceam_yh']*cr_sept_by_yearmonth_fam['ceam_w']
    cr_sept_by_yearmonth_fam['e_aa_w'] = cr_sept_by_yearmonth_fam['err_aa_y']*cr_sept_by_yearmonth_fam['y_w']
    #cr_sept_by_yearmonth_fam['e_ceam_w'] = cr_sept_by_yearmonth_fam['err_ceam_y']*cr_sept_by_yearmonth_fam['y_w']
    cr_sept_by_yearmonth_abc['e_aa_w_yh'] = cr_sept_by_yearmonth_abc['err_aa_yh']*cr_sept_by_yearmonth_abc['aa_w'] # ABC
    #cr_sept_by_yearmonth_abc['e_ceam_w_yh'] = cr_sept_by_yearmonth_abc['err_ceam_yh']*cr_sept_by_yearmonth_abc['ceam_w']
    cr_sept_by_yearmonth_abc['e_aa_w'] = cr_sept_by_yearmonth_abc['err_aa_y']*cr_sept_by_yearmonth_abc['y_w']
    #cr_sept_by_yearmonth_abc['e_ceam_w'] = cr_sept_by_yearmonth_abc['err_ceam_y']*cr_sept_by_yearmonth_abc['y_w']

    month_acc = cr_sept_by_yearmonth.groupby('yearmonth').sum().reset_index()[['yearmonth','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
    #month_acc['acc_ceam_h'] = 100 - month_acc['e_ceam_w_yh']
    month_acc['acc_aa_h'] = round(100 - month_acc['e_aa_w_yh'], 4)
    #month_acc['acc_ceam_y'] = 100 - month_acc['e_ceam_w']
    month_acc['acc_aa_y'] = round(100 - month_acc['e_aa_w'], 4)

    fam_month = cr_sept_by_yearmonth_fam.groupby('familia')[['y','yhat','aa_w','e_aa_w','e_aa_w_yh']].sum().reset_index()[['familia','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
    #fam_month['acc_ceam_h'] = 100 - fam_month['e_ceam_w_yh']
    fam_month['acc_aa_h'] = round(100 - fam_month['e_aa_w_yh'], 4)
    #fam_month['acc_ceam_y'] = 100 - fam_month['e_ceam_w']
    fam_month['acc_aa_y'] = round(100 - fam_month['e_aa_w'], 4)
    fam_month['ym'] = ym
    
    abc_month = cr_sept_by_yearmonth_abc.groupby('CLASS_ABC')[['y','yhat','aa_w','e_aa_w','e_aa_w_yh']].sum().reset_index()[['CLASS_ABC','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
    #abc_month['acc_ceam_h'] = 100 - abc_month['e_ceam_w_yh']
    abc_month['acc_aa_h'] = round(100 - abc_month['e_aa_w_yh'], 4)
    #abc_month['acc_ceam_y'] = 100 - abc_month['e_ceam_w']
    abc_month['acc_aa_y'] = round(100 - abc_month['e_aa_w'], 4)
    abc_month['ym'] = ym
    abc_month.rename(columns={'CLASS_ABC':'abc'},inplace=True)

    final_month_fam = fam_month[['familia','ym','y','yhat','acc_aa_h','acc_aa_y']]
    final_month_abc = abc_month[['abc','ym','y','yhat','acc_aa_h','acc_aa_y']]
    final_month_ym = month_acc[month_acc.yearmonth == ym][['yearmonth','y','yhat','acc_aa_h','acc_aa_y']]
    overall_performance_month = month_acc[['yearmonth','y','yhat','acc_aa_h','acc_aa_y']]

    return final_month_fam, final_month_abc, final_month_ym, serie_temporal, overall_performance_month
