import itertools
from autots import AutoTS

from tsLQC.template_generation import templateGeneration
from tsLQC.backtesting_using_growthrate import backtesting
from tsLQC.constant import params
df = templateGeneration()


def hyperparameter_tuning(ts, backtest_length):
    print("----------------------------------Hyperparameter Tuning ------------------------------------")
    hyperparams_dict = {}

    for validation_points, validation_method in itertools.product(*params.values()):

        try:
            print(validation_points, validation_method)

            model = AutoTS(forecast_length=validation_points,
                           frequency='infer',
                           models_to_validate=0.35,
                           no_negatives=True,
                           ensemble=None,
                           max_generations=5,
                           num_validations=3,
                           validation_method=validation_method,
                           model_list=['ETS'],
                           verbose=0,
                           n_jobs=7)
            model = model.import_template(df, method='only')

            benchmark, forecasted_gr = backtesting(ts, backtest_length, model)
            hyperparams_dict[(validation_points, validation_method)] = abs(benchmark - forecasted_gr)

            print("------------------------------------------------------------")

        except:

            print((validation_points, validation_method), " failed")
            print("------------------------------------------------------------")

    (validation_points, validation_method) = min(hyperparams_dict, key=hyperparams_dict.get)
    print("Best Hyperparameters: ", (validation_points, validation_method))

    return validation_points, validation_method
