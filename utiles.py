## Funciones_Features
import pandas as pd
import numpy as np
import shutil
import os
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class utilities:
    
    @staticmethod
    def generar_senoidales_exogenas(x, top_k=4, extra_steps=0, scale_range=(0, 1), plot=True):
        """
        Genera senoidales exógenas a partir de la descomposición de Fourier.
        
        Params:
        - x: Serie temporal original (1D array)
        - top_k: Número de frecuencias dominantes a usar
        - extra_steps: Cuántos pasos futuros incluir
        - scale_range: Rango para escalar (por defecto [0, 1])
        - plot: Si se desea graficar las senoidales generadas

        Returns:
        - features_df: DataFrame con columnas f_seno_1, f_seno_2, ...
        - t_total: arreglo con los índices de tiempo (original + futuro)
        - info_frec: información de las frecuencias utilizadas
        """
        N = len(x)
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(N)
        amplitudes = np.abs(X)
        half = N // 2
        indices = np.argsort(amplitudes[:half])[::-1][1:top_k+1]  # ignorar k=0

        t_total = np.arange(N + extra_steps)
        x_reconstruido = np.zeros_like(t_total, dtype=np.float64)

        features = []
        info_frec = []

        for idx, k in enumerate(indices):
            freq = freqs[k]
            amp = np.abs(X[k]) / N
            phase = np.angle(X[k])
            wave = amp * np.cos(2 * np.pi * freq * t_total + phase)
            x_reconstruido += wave

            # Escalar la onda
            scaler = MinMaxScaler(feature_range=scale_range)
            wave_scaled = scaler.fit_transform(wave.reshape(-1, 1)).flatten()

            features.append(wave_scaled)
            info_frec.append({
                'feature': f'f_seno_{idx+1}',
                'frecuencia': freq,
                'periodo': 1 / freq if freq != 0 else np.inf,
                'amplitud': amp,
                'fase': phase
            })

        x_reconstruido_scaled = scaler.fit_transform(x_reconstruido.reshape(-1,1)).flatten()   

        features = np.array(features).T
        features_df = pd.DataFrame(features, columns=[f"f_seno_{i+1}" for i in range(top_k)])
        features_df['signal'] = x_reconstruido_scaled
        if plot:
            plt.figure(figsize=(12, 4))
            for i in range(top_k):
                plt.plot(t_total, features[:, i], label=f"Senoidal {i+1}", linestyle="--")
            plt.plot(t_total, features_df['signal'], label='Signal')
            plt.legend()
            plt.grid(True)
            plt.title("Funciones senoidales escaladas (features exógenas)")
            plt.show()

        return features_df, t_total, info_frec
    
    @staticmethod
    def reconstruir_frecuencias(x, top_k=3, extra_steps=0, plot=True):
        """
        Descompone y reconstruye una serie usando las top_k frecuencias más fuertes.
        
        Params:
        - x: Serie temporal (1D array)
        - top_k: Número de frecuencias dominantes a usar
        - extra_steps: Cuántos pasos adicionales generar (predicción futura)
        - plot: Mostrar gráficas

        Returns:
        - t_total: Eje temporal (original + futuros)
        - x_reconstruido: Señal reconstruida con top_k frecuencias
        - info_frec: lista con info de las frecuencias usadas
        - waves: Funciones Sinoidales
        """
        N = len(x)
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(N)

        amplitudes = np.abs(X)
        half = N // 2
        indices = np.argsort(amplitudes[:half])[::-1][1:top_k+1]  # ignorar k=0

        t_total = np.arange(N + extra_steps)
        x_reconstruido = np.zeros_like(t_total, dtype=np.float64)

        info_frec = []
        waves = []
        for k in indices:
            freq = freqs[k]
            amp = np.abs(X[k]) / N
            phase = np.angle(X[k])

            wave = amp * np.cos(2 * np.pi * freq * t_total + phase)
            waves.append(wave)
            x_reconstruido += wave

            info_frec.append({
                'k': k,
                'frecuencia': freq,
                'periodo': 1 / freq if freq != 0 else np.inf,
                'amplitud': amp
            })

        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(np.arange(N), x, label='Original')
            plt.plot(t_total, x_reconstruido, label=f'Reconstrucción con {top_k} frecuencias', linestyle='--')
            if extra_steps:
                plt.axvline(N, color='gray', linestyle=':', label='Inicio del futuro')
            plt.legend()
            plt.grid(True)
            plt.title('Reconstrucción y extrapolación')
            plt.show()

        return t_total, x_reconstruido, info_frec, waves
    
    @staticmethod
    def recursive_forecast(model, seed_sequence, n_steps):
        """
        Genera n_steps hacia adelante a partir de una semilla.
        """
        forecast = []
        input_seq = seed_sequence.copy()

        for _ in range(n_steps):
            pred = model.predict(input_seq[np.newaxis, ..., np.newaxis], verbose=0)[0, 0]
            forecast.append(pred)
            input_seq = np.append(input_seq[1:], pred)  # Shift + append prediction

        return np.array(forecast)

    @staticmethod
    def create_sequences(data, target_col, seq_length):
        """
        data: DataFrame con columnas numéricas y ordenado por fecha
        target_col: nombre de la columna objetivo
        seq_length: número de pasos atrás que el modelo verá
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[target_col].values[i:i + seq_length])
            y.append(data[target_col].values[i + seq_length])
        X = np.array(X)
        y = np.array(y)
        return X[..., np.newaxis], y  # Añadimos dimensión para input_shape (seq_len, 1)

    @staticmethod
    def forecast_unseen(model, input_seq, time_step):
        # Initialize an empty list to store the forecasts
        forecasted_values = []
        
        # Forecast the next 'time_step' number of time steps
        for _ in range(time_step):
            # Make a prediction
            prediction = model.predict(input_seq)
            
            # Append the prediction to the list of forecasted_values
            forecasted_values.append(prediction[0,0])
            
            # Add the prediction to the end of the input sequence
            input_seq = np.append(input_seq, prediction)
            
            # Remove the first value of the input sequence
            input_seq = input_seq[1:]
            
            # Reshape the input sequence to the expected format [samples, time steps, features]
            input_seq = input_seq.reshape((1, len(input_seq), 1))
        
        # Return the list of forecasted values
        return forecasted_values

    @staticmethod
    def escalar_entre_1_y_100(y_series):
        scaler = MinMaxScaler(feature_range=(1, 100))
        y_scaled = scaler.fit_transform(y_series.values.reshape(-1, 1))
        return y_scaled.flatten(), scaler

    @staticmethod
    def desescalar_desde_1_a_100(y_scaled, scaler):
        return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    @staticmethod
    def forecast_unseen(model, input_seq, time_step):
        # Initialize an empty list to store the forecasts
        forecasted_values = []
        
        # Forecast the next 'time_step' number of time steps
        for _ in range(time_step):
            # Make a prediction
            prediction = model.predict(input_seq)
            
            # Append the prediction to the list of forecasted_values
            forecasted_values.append(prediction[0,0])
            
            # Add the prediction to the end of the input sequence
            input_seq = np.append(input_seq, prediction)
            
            # Remove the first value of the input sequence
            input_seq = input_seq[1:]
            
            # Reshape the input sequence to the expected format [samples, time steps, features]
            input_seq = input_seq.reshape((1, len(input_seq), 1))
        
        # Return the list of forecasted values
        return forecasted_values

    @staticmethod
    def generate_quincenal_dates(start_date: str, n_steps: int) -> pd.Series:
        """
        Genera una serie de fechas quincenales (1 y 16 de cada mes) a partir de una fecha inicial.

        Args:
            start_date (str): Fecha inicial en formato 'YYYY-MM-DD'.
            n_steps (int): Número de pasos/quincenas a generar.

        Returns:
            pd.Series: Fechas generadas como objetos datetime.
        """
        start = pd.to_datetime(start_date)
        dates = []
        
        # Asegurarnos de empezar desde el siguiente 1 o 16
        day = start.day
        if day < 16:
            current = start.replace(day=16)
        else:
            current = (start + pd.offsets.MonthBegin(1)).replace(day=1)
        
        for _ in range(n_steps):
            dates.append(current)
            if current.day == 1:
                current = current.replace(day=16)
            else:
                # Si estamos en el día 16, avanzar al día 1 del siguiente mes
                current = (current + pd.offsets.MonthBegin(1)).replace(day=1)
        
        return pd.Series(dates)

    # Function to snap dates to nearest semi-monthly date
    @staticmethod
    def align_to_semi_monthly(date):
        if date.day <= 15:
            return date.replace(day=1)
        elif date.day <= 30:
            return date.replace(day=16)
        else:
            # go to 1st of next month
            next_month = date + pd.offsets.MonthBegin(1)
            return next_month.replace(day=1)

    @staticmethod
    def convertir_fecha_quincenal(quincena_str):
        # Separar Q y el resto
        quincena, mes_ano = quincena_str.split(' ', 1)
        # Traducir meses en español
        meses = {
            'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Ago': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
        }
        mes_abbr, anio = mes_ano.split()
        mes = meses[mes_abbr]

        if quincena == '1Q':
            dia = '01'
        elif quincena == '2Q':
            dia = '16'
        else:
            raise ValueError(f"Formato de quincena no reconocido: {quincena_str}")
        return pd.to_datetime(f"{anio}-{mes}-{dia}")
    
    @staticmethod
    def calcular_inflacion_por_columnas(df, columna_fecha='ds'):
        # Asegurarse que esté ordenado por fecha
        df = df.sort_values(by=columna_fecha).copy()
        
        # Seleccionamos las columnas numéricas (excluyendo la de fecha)
        columnas = df.columns.difference([columna_fecha])
        
        # Calculamos el cambio porcentual para cada una
        for col in columnas:
            nueva_col = f"{col}"
            df[nueva_col] = df[col].pct_change(periods=24) * 100  # cambio porcentual multiplicado por 100
        return df

    @staticmethod
    def add_cyclic_features(df, column, max_val):
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)
        return df
    
    @staticmethod
    def generar_fechas_quincenales(inicio, meses=12):
        fechas = []
        for i in range(meses):
            fecha_mes = inicio + relativedelta(months=i)
            dia1 = pd.Timestamp(fecha_mes.year, fecha_mes.month, 1)
            dia16 = pd.Timestamp(fecha_mes.year, fecha_mes.month, 16)
            fechas.extend([dia1, dia16])
        return pd.DataFrame(fechas, columns=['ds'])

    @staticmethod
    def create_lags(series, n_lags):
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i - n_lags:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    #----------- Forecast n steps ahead -----------
    @staticmethod
    def forecast_future(model, last_window, n_steps, preprocessor=None):
        predictions = []
        input_window = last_window.copy()
        for _ in range(n_steps):
            pred = model.predict(input_window.reshape(1, -1))[0]
            predictions.append(pred)
            input_window = np.roll(input_window, -1)
            input_window[-1] = pred
        if preprocessor:
            predictions = preprocessor.inverse_transform(np.array(predictions))
        return predictions

    @staticmethod
    def clear_folder(folder_path):
        """
        Deletes all files and subdirectories inside the given folder.

        Args:
            folder_path (str): Path to the folder to clear.
        """
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    @staticmethod
    def mape(data=None, yhat=None, y=None):
        diff = abs(data[yhat] - data[y]) / (data[y] + 1e-8)
        if diff > 1:
            return 100
        elif diff < 0:
            return 0
        else:
            return diff * 100
        
    @staticmethod
    def sape(data=None, yhat=None, y=None):
        diff = abs(data[yhat] - data[y]) / (data[yhat] + 1e-8)
        if diff > 1:
            return 100
        elif diff <= 0:
            return 0
        else:
            return diff * 100
    
    @staticmethod
    def data_needed_forecast(data=None, max_encoder_length=None, max_prediction_length=None):
        # select last 24 months from data (max_encoder_length is 24)
        last_known_data = data.copy()
        encoder_data = last_known_data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

        # select last known data point and create decoder data from it by repeating it and incrementing the month
        # in a real world dataset, we should not just forward fill the covariates but specify them to account
        # for changes in special days and prices (which you absolutely should do but we are too lazy here)
        last_data = last_known_data[lambda x: x.time_idx == x.time_idx.max()]
        decoder_data = pd.concat(
            [last_data.assign(PERIODO=lambda x: x.PERIODO + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
            ignore_index=True,
        )

        # add time index consistent with "data"
        decoder_data["time_idx"] = decoder_data["PERIODO"].dt.year * 12 + decoder_data["PERIODO"].dt.month
        decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

        # adjust additional time feature(s)
        decoder_data["month"] = decoder_data.PERIODO.dt.month.astype(str).astype("category")  # categories have be strings

        # combine encoder and decoder data
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
        new_prediction_data['PERIODO'] = new_prediction_data['PERIODO'].apply(lambda x: x.replace(day=15))
        return new_prediction_data
    
    @staticmethod
    def results_dataframe(preds=None, cutoff=None, real=None, presupuesto=None):

        Real_Mensual = real.copy()
        Presupuesto_Mensual =  presupuesto.copy()
        forecasted_tft = preds.output[0][0].cpu().numpy()  # Move to CPU and convert to numpy
        # Start date
        #start_date = '2023-01-15'
        #cutoff = '2022-12-15'
        start_date = pd.to_datetime(cutoff)+ pd.DateOffset(months=1)
        # Generate a date range with 12 periods, say monthly frequency
        date_range = pd.date_range(start=start_date, periods=12, freq='M')  # Monthly frequency ('M')
        # Create a DataFrame
        predictions_dataframe = pd.DataFrame()
        predictions_dataframe['PERIODO'] = date_range
        predictions_dataframe['PERIODO'] = predictions_dataframe['PERIODO'].apply(lambda x: x.replace(day=15))
        predictions_dataframe['Predicted'] = forecasted_tft
        predictions_dataframe['forecast_date'] = start_date
        # Create DataFrame
        evaluation = predictions_dataframe.merge(Real_Mensual[['PERIODO', 'y']], on='PERIODO', how='inner')
        evaluation = evaluation.merge(Presupuesto_Mensual[['PERIODO', 'y']], on='PERIODO', how='inner', suffixes=['_real', '_presupuesto'])
        return evaluation

    @staticmethod
    def laggin_features(n_lags_start=None,
                        n_lags_end=None,
                        data=None,
                        ds=None
                        ):
        #print('Hola')
        #print(data.head())

        for j in range(n_lags_start, n_lags_end+1):
            past_values = []
            for i in range(data.shape[0]):
                last_y_value = data[ data[ds] == (data[ds].iloc[i] - pd.DateOffset(months=j))]['y']
                if last_y_value.empty:
                    past_values.append(0)  # or append `0` if you prefer 0 instead of NaN
                else:
                    past_values.append(last_y_value.values[0])
            data[f'y_last_{j}'] = past_values
        return data

    @staticmethod
    def lags_future_features(dff=None, inference_data = None, n_lags_start=None, n_lags_end = None, ds=None):
        for i in range(n_lags_start,n_lags_end+1):
            dff[f'ds_past_year_{i}'] = dff[ds] - pd.DateOffset(months=i)
            dff = dff.merge(inference_data[[ds, 'y']],#, 'y_last_0', 'y_last_1', 'y_last_2', 'y_last_3', 'y_last_4', 'y_last_5']],#, 'trend', 'cycle']], 
                            left_on=f'ds_past_year_{i}', 
                            right_on=ds,
                            how='left',
                            suffixes=[None, f'_past_{i}']).rename(columns={'y':f'y_last_{i}',
                                                                            #'trend': f'trend_last_{i}',
                                                                            #'cycle':f'cycle_last_{i}'
                                                                            }).drop(columns=[f'ds_past_year_{i}',
                                                                                                                f'{ds}_past_{i}'])
        return dff
    
    @staticmethod
    def split_data_val(data=None, train_years=0, months_val=None, date=None):
        # Ensure data meets the minimum row requirement
        cutoff_val = data[date].max() - pd.DateOffset(months=months_val)
        # Select validation set
        x_val = data[data[date] > cutoff_val]
        # Select training set
        if train_years == 0:
            x_train = data[data[date] <= cutoff_val]
        else:
            cutoff_train = data[date].max() - pd.DateOffset(years=train_years)
            x_train = data[(data[date] >= cutoff_train) & (data[date] <= cutoff_val)]
        return x_train, x_val
    
    @staticmethod
    def split_data_val_months(data=None, meses_train=None, meses_val=None, date='ds'):
        """
        Divide los datos en conjuntos de entrenamiento y validación, usando los últimos (meses_train + meses_val) meses.
        
        Parámetros:
        - data: DataFrame con una columna de fechas
        - meses_train: número de meses para entrenamiento
        - meses_val: número de meses para validación
        - date: nombre de la columna de fechas (por defecto 'ds')

        Retorna:
        - x_train: últimos meses_train meses antes de validación
        - x_val: últimos meses_val meses
        """
        data = data.copy()
        data[date] = pd.to_datetime(data[date])
        cutoff_total = data[date].max() - pd.DateOffset(months=(meses_train + meses_val))
        cutoff_val = data[date].max() - pd.DateOffset(months=meses_val)

        data_recent = data[data[date] > cutoff_total]
        x_train = data_recent[data_recent[date] < cutoff_val]
        x_val = data_recent[data_recent[date] >= cutoff_val]
        
        return x_train, x_val
    
    @staticmethod
    def features_from_date(data=None, ds=None):
        data['quarter'] = data[ds].dt.quarter
        data['month'] = data[ds].dt.month
        data['year'] = data[ds].dt.year
        data['dayofyear'] = data[ds].dt.dayofyear
        data['dayofmonth'] = data[ds].dt.day
        data['weekofyear'] = data[ds].dt.isocalendar().week 
        return data.sort_values(by=ds)
    
    @staticmethod # its the same ahora que me doy cuenta jhaha
    def features_neural_networks(data=None, ds=None):
        data['quarter'] = data[ds].dt.quarter
        data['month'] = data[ds].dt.month
        data['year'] = data[ds].dt.year
        data['dayofyear'] = data[ds].dt.dayofyear
        data['dayofmonth'] = data[ds].dt.day
        data['weekofyear'] = data[ds].dt.isocalendar().week 
        return data.sort_values(by=ds)

    def metricas_analytics_neural_obj(self, cr, cr_sept):
        cr_sept['yhat'] = cr_sept['yhat'].astype(float).astype(int)
        cr_sept['ds'] = pd.to_datetime(cr_sept['ds'])
        #Data
        cr['ds'] = pd.to_datetime(cr['ds'])
        cr = cr.set_index('ds').groupby(['unique_id','familia','CLASS_ABC']).resample('W-Mon')['y'].sum().reset_index()
        cr_sept_merge = cr_sept.merge(cr,on=['ds','unique_id'],how='inner')   # Se une el Forecast con los Datos Reales.
        cr_sept_merge['yearmonth'] = cr_sept_merge['ds'].apply(lambda x: int(str(x).replace('-', '')[: 6])) # Se crea la variable Yearmonth
        cr_sept_by_yearmonth = cr_sept_merge.set_index('ds').groupby(['yearmonth','unique_id','familia','CLASS_ABC']).sum().reset_index()# Se agrupa por Yearmonth
        cr_sept_by_yearmonth['err_aa_y'] = cr_sept_by_yearmonth.apply(self.mape , args=('yhat', 'y'), axis=1)  # Calculo de Mape y
        cr_sept_by_yearmonth['err_aa_yh'] = cr_sept_by_yearmonth.apply(self.sape , args=('yhat', 'y'), axis=1) # Calculo de Mape yhat
        cr_sept_by_yearmonth['acc_sku_aa_y'] = 100 - cr_sept_by_yearmonth['err_aa_y'].round(2) ## Asertvidad por SKU - Mape Y
        cr_sept_by_yearmonth['acc_sku_aa_yhat'] = 100 - cr_sept_by_yearmonth['err_aa_yh'].round(2) ## Asertvidad por SKU - Mape Yhat
        y_acc = cr_sept_by_yearmonth.groupby('yearmonth')['y'].sum().reset_index().rename(columns={'y':'y_acc'}) # Acumulado de venta Mensual
        yhat_acc = cr_sept_by_yearmonth.groupby('yearmonth')['yhat'].sum().reset_index().rename(columns={'yhat':'aa'}) # Acumulado de Forecasts
        # Union de DataFrames con Sus Respectivos Acumulados, Familia, ABC y SKU
        cr_sept_by_yearmonth = cr_sept_by_yearmonth.merge(y_acc,on='yearmonth',how='left') # Se unen los acumulados
        cr_sept_by_yearmonth = cr_sept_by_yearmonth.merge(yhat_acc,on='yearmonth',how='left')
        #Calculo de Weights in Family, ABC and SKU.
        cr_sept_by_yearmonth['y_w'] = cr_sept_by_yearmonth['y']/cr_sept_by_yearmonth['y_acc'] # SKU
        cr_sept_by_yearmonth['aa_w'] =  cr_sept_by_yearmonth['yhat']/cr_sept_by_yearmonth['aa'] # Weights based on the forecasted of analytics
        # Multiplicacion de Errores por los Weights de cada SKU de la Venta Total de cada CLASE.
        cr_sept_by_yearmonth['e_aa_w_yh'] = cr_sept_by_yearmonth['err_aa_yh']*cr_sept_by_yearmonth['aa_w'] 
        cr_sept_by_yearmonth['e_aa_w'] = cr_sept_by_yearmonth['err_aa_y']*cr_sept_by_yearmonth['y_w']
        month_acc = cr_sept_by_yearmonth.groupby('yearmonth').sum().reset_index()[['yearmonth','y','yhat','aa_w','e_aa_w','e_aa_w_yh']]
        month_acc['acc_aa_h'] = 100 - month_acc['e_aa_w_yh'].round(2)
        month_acc['acc_aa_y'] = 100 - month_acc['e_aa_w'].round(2)
        overall_performance_month = month_acc[['yearmonth','y','yhat','acc_aa_h','acc_aa_y']]
        return overall_performance_month