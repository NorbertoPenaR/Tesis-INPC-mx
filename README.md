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
```

La corrida `--smoke` comprueba la integración con pocas épocas y un horizonte
reducido. No sustituye el benchmark *rolling-origin* de 6 y 12 meses; por ello,
sus resultados no se presentan como evidencia final en la tesis.

Las fuentes de la tesis se encuentran en `thesis/` y el PDF compilado en
`output/pdf/Tesis_final_HP_PyTorch.pdf`.
