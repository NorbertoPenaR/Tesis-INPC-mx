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
