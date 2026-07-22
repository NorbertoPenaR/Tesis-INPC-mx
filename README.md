# 📊 Prediccón de la Inflación y el Índice Nacional de Precios al Consumidor y sus Componentes en México
## Un estudio comparativo entre modelos clásicos, árboles de decisión y redes neuronales

Este proyecto forma parte de la tesis de licenciatura en Matemáticas Aplicadas. Su objetivo es predecir la inflación en México mediante un enfoque comparativo que evalúa modelos clásicos, algoritmos de árboles de decisión y redes neuronales profundas.

## 📌 Objetivo

Evaluar el desempeño de distintos modelos de predicción de series temporales para estimar la inflación en México, usando métricas como RMSE y MAPE, tanto en horizontes cortos como largos.

## 🧠 Modelos Implementados

- Naive
- Suavizado Exponencial Triple (Holt-Winters)
- XGBoost
- Redes Neuronales Recurrentes (RNN)
- Long Short-Term Memory (LSTM)
- DeepAR
- Transformer
- D³VAE

## 🧾 Estructura del Proyecto

- `obj.py`: Definición de las funciones objetivo de cada modelo para obtener los mejores parametros dado el conjunto de prueba.
- `fit.py`: Entrenamiento de modelos.
- `predict.py`: Predicciones a futuro, y resultados sobre datos no vistos para evaluar el rendimiento de los modelos.
- `utils.py`: Funciones auxiliares. 
- `process_data.py`: Limpieza, transformación y extracción de datos
- `main.py`: Orquestador principal del flujo completo

## 📁 Datos

Los datos se obtuvieron del INEGI y consisten en series quincenales del Índice Nacional de Precios al Consumidor (INPC) y sus componentes.
https://www.inegi.org.mx/temas/inpc/

## ▶️ Ejecución

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate   # en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el flujo principal
python src/main.py
```

## Extensión HP con modelos PyTorch propios

El paquete `inpc_forecasting` agrega un flujo reproducible que descompone cada
serie como `y = tendencia + ciclo`, entrena instancias independientes para
ambos componentes y reconstruye el pronóstico final. Incluye implementaciones
propias de RNN, LSTM, DeepAR y Transformer. El código histórico basado en
NeuralForecast se conserva, pero no es una dependencia del paquete nuevo.

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m inpc_forecasting.cli --config configs/hp_pytorch.yaml --smoke
python -m inpc_forecasting.cli --config configs/hp_pytorch.yaml --rolling
python -m inpc_forecasting.cli --config configs/hp_pytorch.yaml --rolling `
  --trend-transform none --output-dir outputs/hp_none
python -m inpc_forecasting.cli --config configs/hp_pytorch.yaml --rolling `
  --trend-transform logp1 --output-dir outputs/hp_logp1
python scripts/analyze_hp_results.py `
  --predictions outputs/hp_pytorch/rolling_predictions.csv `
  --data data/ca56_2018a-2025_10_14.csv `
  --output-dir outputs/hp_pytorch
```

La corrida `--smoke` comprueba la integración con pocas épocas y un horizonte
reducido. No sustituye el benchmark *rolling-origin* de 6 y 12 meses; por ello,
sus resultados no se presentan como evidencia final en la tesis.

El benchmark verifica antes de entrenar que cada corte tenga todos los valores
reales exigidos por el horizonte máximo. El script de análisis audita valores
faltantes, infinitos y duplicados, y compara las redes contra persistencia y
contra el pronóstico aislado de la tendencia.

Las columnas `y_true_trend` y `y_true_cycle` son etiquetas retrospectivas: se
calculan con HP sobre entrenamiento más el horizonte observado únicamente
después de producir el pronóstico. Sirven para medir `mae_trend` y `mae_cycle`,
pero nunca entran al entrenamiento. La CLI acepta `--trend-model` y
`--cycle-model` para combinar arquitecturas distintas; los scripts
`combine_component_forecasts.py` y `evaluate_trend_baselines.py` permiten
comparar combinaciones y tendencias HP analíticas sin reentrenar el ciclo.

Las fuentes de la tesis se encuentran en `thesis/` y el PDF compilado en
`output/pdf/Tesis_final_HP_PyTorch.pdf`.
