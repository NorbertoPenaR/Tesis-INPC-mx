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

path_experimentos = 'C:/Users/betos/OneDrive/Desktop/tesis_code/ray_experiments'

def short_trial_name(trial):
    return f"trial_{trial.trial_id[:6]}"
# Clasical Models

# Holt Winters
def fit_holt_winters(data=None, cutoff_date=None):
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    def short_trial_name(trial):
        return f"trial_{trial.trial_id[:6]}"

    holt_param_space = {
        # Holt Parameters
        "trend_type": tune.choice(["mul", "add"]),
        "seasonal_type": tune.choice(["mul", "add"]),
        "damped_trend": tune.choice([True, False]),
        "use_boxcox": tune.choice([True, False]),
        # Data Splitting Parameters
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4),
        #"seasonal_periods": tune.randint(12, 24),
        # The right seasonal period is achieved by applying 
        # the Discrete Fourier Transformation to our
        # Target Variable.
    }

    hyperopt_search = HyperOptSearch(holt_param_space, metric="rmse", mode="min")

    tuner = tune.Tuner(
        trainable=partial(obj.obj_holt_winters, data=data),
        tune_config=tune.TuneConfig(
            num_samples=40,
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name,# Nombre de los Trials. 
        ),
        run_config=tune.RunConfig(
            storage_path="./ray_experiments"# Almacenamiento de los Experimentos 
                                            # Paramétricos
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]

# XGBoost
def fit_xgb(data=None, cutoff_date=None):
    xgb_params = {
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4),
        'max_depth':tune.randint(2, 35),
        'colsample_bytree':tune.uniform(.5,1),
        'subsample':tune.uniform(.5,.95),
        'alpha':tune.uniform(0,0),
        'eta':tune.uniform(.1,.4),
        'lambdaa':tune.uniform(.5,3),
        'num_boost_round':tune.randint(50, 150),
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]
    hyperopt_search = HyperOptSearch(xgb_params, metric="rmse", mode="min")
    # Se establece el algoritmo de busqueda paramétrico
    # (Bayesian Optimization)
    # Se usaría otra manera, pero dado que se trata de cosas más especificas
    # Se opta por usar Tune. ¿Optuna? ¿?
    xgb_tuner = tune.Tuner(
        trainable=partial(obj.obj_xgb, data=data),
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            search_alg=hyperopt_search
        ),
        run_config=tune.RunConfig(
            storage_path="./ray_experiments"# Almacenamiento de los Experimentos 
                                            # Paramétricos
        ),
    )
    results = xgb_tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]

# RNN
def fit_rnn(data=None, cutoff_date=None, iteraciones=None):
    rnn_params = {
        # Data Splitting Params
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4),
        # Neural Network Parameters
        'h':tune.randint(40,40),
        'input_size':tune.randint(1,8),
        'neurons':tune.choice([16, 32, 64, 128, 256]),
        'layers':tune.randint(1,8),
        "max_steps": tune.quniform(lower=100, upper=2500, q=100),
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(rnn_params, metric="rmse", mode="min")

    rnn_tuner = tune.Tuner(
        partial(obj.obj_rnn, data=data),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
        storage_path=path_experimentos,
        ),
    )
    results = rnn_tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]

# LSTM
def fit_lstm(data=None, cutoff_date=None, iteraciones=None):
    lstm_params = {
        # Data Splitting Params
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4),
        # Neural Network Parameters
        'h':tune.choice([40,42]),
        'input_size':tune.randint(3,8),
        'layers':tune.randint(2,8),
        "max_steps": tune.quniform(lower=1000, upper=2500, q=100),
        'neurons':tune.choice([32, 64, 128, 256]),
        'learning_rate':tune.qloguniform(1e-4, 1e-1, 5e-5),
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(lstm_params, metric="rmse", mode="min")

    lstm_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_lstm, data=data),
            {"cpu": 6, "gpu": 1}  # 👈 declare GPU use here
        ),
        #partial(obj.obj_lstm, data=data),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
        storage_path=path_experimentos,
        ),
    )
    
    results = lstm_tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]

# Deep Ar
def fit_deep_ar(data=None, cutoff_date=None, iteraciones=None):
    lstm_params = {
        # Data Splitting Params.
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4), # 2 a 4 meses
        # Implica tener un horizonte mayor a 2 y 4 meses.
        # Es decir, Dado el Máximo Rango
        # Tomar ese como punto de partida para el resto de experimentos que se harán. 
        # Entonces, se requieren 4*4=16 Es decir, para evaluar en datos de test
        # Se necesitarán 16+4 = 20 
        # 20*2 = 40; El doble. 
        # Neural Network Params
        'h':tune.randint(40,40),
        'input_size':tune.randint(1, 8),
        'layers':tune.randint(1,8),
        'trajectories':tune.randint(50, 150),
        'learning_rate':tune.qloguniform(1e-4, 1e-1, 5e-5),
        "max_steps": tune.quniform(lower=100, upper=2500, q=100),
        'neurons':tune.choice([16, 32, 64, 128, 256])
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(lstm_params, metric="rmse", mode="min")
    DeepAr_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_deep_ar, data=data),
            {"cpu": 6, "gpu": 1}  # 👈 declare GPU use here
        ),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
        storage_path=path_experimentos,
        ),
    )
    results = DeepAr_tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]

# Transformers
# De acuerdo con un paper, lo ideal es comenzar con poco
# Encontrar el sweet spot. Entre más complejo sea un modelo
# La probabilidad de que este se overfitee es mayor. 
# Haremos dos experimentos. 
# Uno consistirá en incluir las variables exogenas temporales, 
# senoidales y el componente de la señal
def fit_transformer(data=None, cutoff_date=None, iteraciones=None):
    transformer_params = {
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4),
        # Params
        'h':tune.randint(40,40),
        'input_size':tune.randint(1, 8),
        'neurons':tune.choice([16, 32, 64, 128, 256]),
        'conv_size':tune.choice([16, 32, 64]),
        'n_heads':tune.randint(2, 8),
        "max_steps": tune.quniform(lower=500, upper=2500, q=100),
        'neurons':tune.choice([16, 32, 64, 128, 256])
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(transformer_params, metric="rmse", mode="min")

    transformer_tuner = tune.Tuner(
        tune.with_resources(
            partial(obj.obj_transformer, data=data),
            {"cpu": 6, "gpu": 1}  # 👈 declare GPU use here
        ),

        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
        storage_path=path_experimentos,
        ),
    )
    results = transformer_tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]
 
# NHITS
def fit_nhits(data=None, cutoff_date=None, iteraciones=None):
    nhits_params={
        # Split Data Params
        'years':tune.randint(2, 8),
        'months':tune.randint(2, 4),
        # Neural Network Parameters
        'h':tune.choice([40]),
        'input_size':tune.randint(1, 8),
        'neurons':tune.choice([16, 32, 64, 128, 256]),
        "max_steps": tune.quniform(lower=500, upper=2500, q=100),
        "n_pool_kernel_size": tune.choice(
            [[2,1,1], [2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [2,1,1],
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-4, 1e-1),         
    }
    # Ingesta de Datos con Fecha de Corte
    data['ds'] = pd.to_datetime(data['ds'])
    data = data[data['ds']<=cutoff_date]

    hyperopt_search = HyperOptSearch(nhits_params, metric="rmse", mode="min")

    nhits_tuner = tune.Tuner(
        # Es necesario establecer 
        tune.with_resources(
            partial(obj.obj_nhits, data=data),
            {"cpu": 6, "gpu": 1}  # 👈 declare GPU use here
        ),
        
        tune_config=tune.TuneConfig(
            num_samples=iteraciones,
            scheduler=ASHAScheduler(metric="rmse", mode="min"),
            search_alg=hyperopt_search,
            trial_dirname_creator=short_trial_name  #trial folder name
        ),
        run_config=tune.RunConfig(
        storage_path=path_experimentos,
        ),
    )
    results = nhits_tuner.fit()
    best_result = results.get_best_result(metric="rmse", mode="min")
    print("Best RMSE:", best_result.metrics["rmse"])
    print("Best config:", best_result.config)
    #best_result.
    # Config y Mejor Resultado
    return best_result.config, best_result.metrics["rmse"]

# DVAE
  