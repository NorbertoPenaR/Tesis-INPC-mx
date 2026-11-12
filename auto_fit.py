from functools import partial
from bayes_opt import BayesianOptimization
import best_model_dist
# sellin_gt_20250731_1129
# sellin_cr_20250731_1129
# sellin_ni_20250731_1129
# sellin_pa_20250731_1129
# sellin_do_20250731_1129
# 


##

#
def obj_dist(ice_cold=None, alpha_v1=None):
    a, b, c, d = best_model_dist.simulacion(country= 'mex',
                                        date_file='20250801_0914',
                                        i_d=ice_cold,
                                        clases=['A'],
                                        alpha_v3=alpha_v1,
                                        month_list=['04', '05', '06'],
                                        year_months=[202505, 202506, 202507]
                                        )
    
    Error_mape = b['acc_aa_y'].mean()
    Error_sape = b['acc_aa_h'].mean()
    Error_mitigado = (Error_mape+Error_sape)/2
    return Error_mitigado


def fit_dist():
# train model
    pbounds = {
        'ice_cold':(.7,1.2),
        'alpha_v1':(.01, .3)
    }

    # try:
    #xgb_partial = partial(obj_dist, data=None)
    
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimization(f=obj_dist,
                                    pbounds=pbounds,
                                    random_state=42,
                                    verbose=2,
                                    allow_duplicate_points=False)

    # Perform Bayesian optimization
    optimizer.maximize(init_points=5, n_iter=10) # 15 ,  40

    dic_params = dict()
    #dic_params[''] = category
    dic_params = optimizer.max['params']
    dic_params['accuracy'] = optimizer.max['target']
    print(dic_params)
    return dic_params

p = fit_dist()
print(p)