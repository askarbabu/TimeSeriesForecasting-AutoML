import itertools
from dataclasses import dataclass
from autots import AutoTS

from tsLQC.template_generation import template_generation
from tsLQC.backtesting_using_growthrate import backtesting
from tsLQC.constant import params, n_jobs, verbose, frequency, hp_tuning_models_to_validate, \
    hp_tuning_max_generations, hp_tuning_num_validations, hp_tuning_model_list
df = template_generation()


@dataclass
class ModelValuation:
    validation_points: float
    validation_method: str
    accuracy: float


def hyperparameter_tuning(ts, backtest_length):
    valuations = [backtesting_models(backtest_length, ts, validation_method, validation_points) for
                  validation_points, validation_method in itertools.product(*params.values())]
    best_params = min(valuations, key=lambda x: x.accuracy)
    return best_params.validation_points, best_params.validation_method


def backtesting_models(backtest_length, ts, validation_method, validation_points):
    try:
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
        accuracy = abs(benchmark - forecasted_gr)
        return ModelValuation(validation_points, validation_method, accuracy)
    finally:
        pass
