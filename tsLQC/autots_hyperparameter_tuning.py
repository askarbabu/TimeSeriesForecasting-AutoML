import logging
import itertools
from dataclasses import dataclass
from autots import AutoTS
from pandas import Series
from typing import Tuple

from tsLQC.template_generation import template_generation
from tsLQC.backtesting_using_growthrate import backtesting
from tsLQC.constant import N_JOBS, VERBOSE, FREQUENCY

PARAMS = {'validation_points': [4, 6, 8, 10], 'validation_method': ['backward', 'even']}
HP_TUNING_MODELS_TO_VALIDATE = 0.35
HP_TUNING_MAX_GENERATIONS = 5
HP_TUNING_NUM_VALIDATIONS = 3
HP_TUNING_MODEL_LIST = ['ETS']

df = template_generation()


@dataclass
class ModelValuation:
    validation_points: int
    validation_method: str
    accuracy: float


def hyperparameter_tuning(ts: Series, backtest_length: int) -> Tuple[int, str]:
    valuations = [backtesting_models(backtest_length, ts, validation_method, validation_points) for
                  validation_points, validation_method in itertools.product(*PARAMS.values())]
    best_params = min(valuations, key=lambda x: x.accuracy)
    logging.info(f'Best Hyperparameters: {(best_params.validation_points, best_params.validation_method)}')
    return best_params.validation_points, best_params.validation_method


def backtesting_models(backtest_length: int, ts: Series, validation_method: str, validation_points: int) -> \
        ModelValuation:
    try:
        logging.info(f'Running {(validation_points, validation_method)} ....')
        model = AutoTS(forecast_length=validation_points,
                       frequency=FREQUENCY,
                       models_to_validate=HP_TUNING_MODELS_TO_VALIDATE,
                       no_negatives=True,
                       ensemble=None,
                       max_generations=HP_TUNING_MAX_GENERATIONS,
                       num_validations=HP_TUNING_NUM_VALIDATIONS,
                       validation_method=validation_method,
                       model_list=HP_TUNING_MODEL_LIST,
                       verbose=VERBOSE,
                       n_jobs=N_JOBS)
        model = model.import_template(df, method='only')

        benchmark, forecasted_gr = backtesting(ts, backtest_length, model)
        accuracy = abs(benchmark - forecasted_gr)
        return ModelValuation(validation_points, validation_method, accuracy)
    except:
        pass
