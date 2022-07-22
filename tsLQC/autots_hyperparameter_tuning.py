import itertools
from autots import AutoTS

from tsLQC.template_generation import templateGeneration
from tsLQC.backtesting_using_growthrate import backtesting
from tsLQC.constant import params, n_jobs, verbose,  frequency, hp_tuning_models_to_validate, \
    hp_tuning_max_generations, hp_tuning_num_validations, hp_tuning_model_list
df = templateGeneration()


def hyperparameter_tuning(ts, backtest_length):
    print("----------------------------------Hyperparameter Tuning ------------------------------------")
    hyperparams_dict = {}

    for validation_points, validation_method in itertools.product(*params.values()):

        try:
            print(validation_points, validation_method)

            model = AutoTS(forecast_length=validation_points,
                           frequency=frequency,
                           models_to_validate=hp_tuning_models_to_validate,
                           no_negatives=True,
                           ensemble=None,
                           max_generations=hp_tuning_max_generations,
                           num_validations=hp_tuning_num_validations,
                           validation_method=validation_method,
                           model_list=hp_tuning_model_list,
                           verbose=verbose,
                           n_jobs=n_jobs)
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
