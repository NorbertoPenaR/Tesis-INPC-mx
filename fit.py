# fit 
from ray import tune
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from utiles import utilities
from ray.tune.schedulers import ASHAScheduler
import obj
from functools import partial
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, MAE, RMSE, MAPE
from ray.tune.search import ConcurrencyLimiter

from ray.tune.search.bayesopt import BayesOptSearch
import ray

from ray.tune.logger import CSVLoggerCallback
from ray.tune import TuneConfig, RunConfig, Tuner

from ray.tune.logger import NoopLogger
import os

# Silenciar el logger de ray.tune
import logging
#logging.getLogger("ray.tune.search.hyperopt.hyperopt_search").setLevel(logging.ERROR)

# Para silenciar logs that seem to be completly useless. 
# I dont see the point, just visual noise on the console; thats what they are. 
def silent_logger_creator(config):
    logdir = os.path.join(path_experimentos, "silent_logs")
    os.makedirs(logdir, exist_ok=True)
    return NoopLogger(config, logdir)

ray.init(log_to_driver=False)


path_experimentos = 'C:/Users/betos/OneDrive/Desktop/tesis_code/ray_experiments'

def short_trial_name(trial):
    return f"trial_{trial.trial_id[:6]}"
# Clasical Models

# Holt Winters
def fit_holt_winters(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                    Metric=None, horizon=None, Mes_val=None):
    holt_param_space = {
        # Holt Parameters
        "trend_type": tune.choice(["mul", "add"]),
        "seasonal_type": tune.choice(["mul", "add"]),
        "damped_trend": tune.choice([True, False]),
        "use_boxcox": tune.choice([True, False]),
        # Data Splitting Parameters
        'years':tune.randint(5, 8),
        'months':tune.choice([Mes_val]),
        # Metric
        'metric': tune.choice([Metric]),
        # Future Steps
        'h':tune.choice([horizon]),
        "seasonal_periods": tune.randint(12, 64),
        # The right seasonal period is achieved by applying 
        # the Discrete Fourier Transformation to our
        # Target Variable.
        # Frequency
        'freq': tune.choice([freak])
    }
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    hyperopt_search = HyperOptSearch(holt_param_space, metric="error", mode="min")

    hw_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_holt_winters, data=data),
            {"cpu": 12, "gpu": 1}  # 👈 declare GPU use here
        ),
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="hw_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )

    results = hw_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]

# XGBoost
def fit_xgb(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                    Metric=None, horizon=None, Mes_val=None):
    xgb_params = {
        'years':tune.randint(6, 8),
        'months':tune.choice([Mes_val]),
        # XGB Params
        'max_depth':tune.randint(2, 45),
        'colsample_bytree':tune.uniform(.5,1),
        'subsample':tune.uniform(.5,.95),
        'alpha':tune.uniform(0,4),
        'eta':tune.uniform(.1,.4),
        'lambdaa':tune.uniform(.5,3),
        'num_boost_round':tune.randint(50, 150),
        # Frequency
        #'freq': tune.choice([freak])
        # Metric
        'metric': tune.choice([Metric]),
        # Signals
        'signals':tune.randint(4, 30),
        # Future Steps
        'h':tune.choice([horizon]),
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    hyperopt_search = HyperOptSearch(xgb_params, metric="error", mode="min")
    # Se establece el algoritmo de busqueda paramétrico
    # (Bayesian Optimization)
    # Se usaría otra manera, pero dado que se trata de cosas más especificas
    # Se opta por usar Tune. ¿Optuna? ¿?
    xgb_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_xgb, data=data),
            {"cpu": 12, "gpu": 1}  # 👈 declare GPU use here
        ),
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="xgb_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )

    results = xgb_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]

# RNN
def fit_rnn(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None):
    rnn_params = {
        # Data Splitting Params
        'years':tune.randint(2, 8),
        'months':tune.choice([Mes_val]),
        # Future Steps
        'h':tune.choice([horizon]),
        # Neural Network Parameters
        'input_size':tune.randint(1,8),
        'neurons':tune.choice([16, 32, 64, 128, 256]),
        'layers':tune.randint(1,8),
        "max_steps": tune.quniform(lower=100, upper=2000, q=100),
        # Frequency
        'freq': tune.choice([freak]),
        # Metric
        'metric': tune.choice([Metric])
    }

    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(rnn_params, metric="error", mode="min")

    rnn_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_rnn, data=data),
            {"cpu": 12, "gpu": 1}  # 👈 declare GPU use here
        ),
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="rnn_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )
    results = rnn_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    #print("Best RMSE:", best_result.metrics["rmse"])
    #print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]

# LSTM
def fit_lstm(data=None, cutoff_date=None, iteraciones=None, freak=None, 
            Metric=None, horizon=None, Mes_val=None):
    lstm_params = {
        # Data Splitting Params
        'years':tune.randint(3, 8),
        #'months':tune.randint(4, 5),
        'months':tune.choice([Mes_val]),
        # Future Steps
        'h':tune.choice([horizon]),
        # Neural Network Parameters
        'input_size':tune.randint(1,8),
        'layers':tune.randint(2,8),
        "max_steps": tune.quniform(lower=1000, upper=1500, q=100),
        'neurons':tune.choice([32, 64, 128, 256]),
        'learning_rate':tune.choice([0.001]),
        # Frequency
        'freq': tune.choice([freak]),
        # Metric
        'metric': tune.choice([Metric])
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(lstm_params, metric="error", mode="min")

    lstm_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_lstm, data=data),
            {"cpu": 12, "gpu": 1}  # 👈 declare GPU use here
        ),
        #partial(obj.obj_lstm, data=data),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="lstm_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )
    results = lstm_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]

# Deep Ar
def fit_deep_ar(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                Metric=None, horizon=None, Mes_val=None):
    deep_ar_params = {
        # Data Splitting Params.
        'years':tune.randint(4, 8),
        'months':tune.choice([Mes_val]), # 2 a 4 meses
        # Implica tener un horizonte mayor a 2 y 4 meses.
        # Es decir, Dado el Máximo Rango
        # Tomar ese como punto de partida para el resto de experimentos que se harán. 
        # Entonces, se requieren 4*4=16 Es decir, para evaluar en datos de test
        # Se necesitarán 16+4 = 20 
        # 20*2 = 40; El doble.
        # Future Steps
        'h':tune.choice([horizon]),
        # Neural Network Params
        'input_size':tune.randint(1, 6),
        'layers':tune.randint(1,6),
        'trajectories':tune.randint(50, 150),
        'learning_rate':tune.qloguniform(1e-4, 1e-1, 5e-5),
        "max_steps": tune.quniform(lower=100, upper=2500, q=100),
        'neurons':tune.choice([16, 32, 64, 128]),
        # Frequency
        'freq': tune.choice([freak]),
        # Metric
        'metric': tune.choice([Metric])
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(deep_ar_params, metric="error", mode="min")
    DeepAr_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_deep_ar, data=data),
            {"cpu": 12, "gpu": 1}  # 👈 declare GPU use here
        ),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="deepAr_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )
    results = DeepAr_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]

# Transformers
# De acuerdo con un paper, lo ideal es comenzar con poco
# Encontrar el sweet spot. Entre más complejo sea un modelo
# La probabilidad de que este se overfitee es mayor. 
# Haremos dos experimentos. 
# Uno consistirá en incluir las variables exogenas temporales, 
# senoidales y el componente de la señal

def fit_transformer(data=None, cutoff_date=None, iteraciones=None, freak=None, 
                    Metric=None, horizon=None, Mes_val=None):
    transformer_params = {
        'years':tune.randint(2, 8),
        'months':tune.choice([Mes_val]),
        # Future Steps
        'h':tune.choice([horizon]),
        # Params
        'input_size':tune.randint(1, 8),
        'neurons':tune.choice([16, 32, 64, 128]),
        'conv_size':tune.choice([16, 32]),
        'n_heads':tune.randint(2, 8),
        "max_steps": tune.quniform(lower=500, upper=2000, q=100),
        # Frequency
        'freq': tune.choice([freak]),
        # Metric
        'metric': tune.choice([Metric])
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(transformer_params, metric="error", mode="min")

    transformer_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_transformer, data=data),
            {"cpu": 12, "gpu": 1}  # Se utilizaran 12 cpu y 1 gpu 
        ),
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="transformer_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )
    results = transformer_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]
 
# NHITS
def fit_nhits(data=None, cutoff_date=None, iteraciones=None, freak=None, 
              Metric=None, horizon=None, Mes_val=None):
    nhits_params={
        # Split Data Params
        'years':tune.randint(3, 8),
        'months':tune.choice([Mes_val]),
        # Future Steps
        'h':tune.choice([horizon]),
        # Neural Network Parameters
        'input_size':tune.randint(1, 6),
        'neurons':tune.choice([128, 256, 512, 1024]),
        "max_steps": tune.quniform(lower=500, upper=2500, q=100),
        "n_pool_kernel_size": tune.choice([3 * [2], 3 * [4], 3 * [8], [8, 4, 1], [16, 8, 1]]),
        "n_freq_downsample": tune.choice([[168, 24, 1],
                                        [24, 12, 1],
                                        [180, 60, 1],
                                        [60, 8, 1],
                                        [40, 20, 1]]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        # Frequency
        'freq': tune.choice([freak]),
        # Metric
        'metric': tune.choice([Metric])
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(nhits_params, metric="error", mode="min")

    nhits_tuner = tune.Tuner(
        # Es necesario establecer 
        tune.with_resources(
            partial(obj.obj_nhits, data=data),
            {"cpu": 12, "gpu": 1}  # Recursos a ser usados
            # Tengo un total de 24 cpu's.
            # Entonces le dedicaré 10 para acelerar el experimento.
        ),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="error", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
            verbose=0,
            name="nhits_fast",
            log_to_file=False,
            storage_path=path_experimentos,
            callbacks=[CSVLoggerCallback()]
        )
    )
    results = nhits_tuner.fit()
    best_result = results.get_best_result(metric="error", mode="min")
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["error"]

# DVAE
  