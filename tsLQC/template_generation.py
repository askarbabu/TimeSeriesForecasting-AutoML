import itertools
import json
import pandas as pd
from typing import Tuple
from autots import AutoTS
from autots.models.ensemble import EnsembleTemplateGenerator
from tsLQC.constant import METRIC_WEIGHTING, VERBOSE, DATE_COL,\
    VALUE_COL, N_JOBS, FREQUENCY, MODELS_TO_VALIDATE, NO_NEGATIVES


f = open('model_params.json', "r")
params_dict = json.loads(f.read())


def generatetemplate(modelname: str, params: dict) -> pd.DataFrame:

    d = {}
    paramslst = []
    my_dict = {}
    for x in list(itertools.product(*params.values())):
        for i in range(len(x)):
            my_dict[list(params.keys())[i]] = x[i]
        paramslst.append(json.dumps(my_dict))
    d['Model'] = [modelname]*len(paramslst)
    d['ModelParameters'] = paramslst
    d['TransformationParameters'] = ['''{"fillna": "fake_date", "transformations": {}, "transformation_params":{}}'''] \
        * len(paramslst)
    d['Ensemble'] = [0] * len(paramslst)

    return pd.DataFrame(d)


def template_generation() -> pd.DataFrame:
    temp_df = pd.DataFrame()

    for i in params_dict.keys():
        temp_df = pd.concat([temp_df, generatetemplate(i, params_dict[i])])

    df = temp_df.reset_index().rename(columns={'index': 'ID'})

    return df


def generate_ensemble_models(ts: pd.Series, model: AutoTS, best_simple_models: pd.DataFrame) ->\
        Tuple[AutoTS, pd.DataFrame]:

    model.initial_results.model_results = best_simple_models
    ens_temp = EnsembleTemplateGenerator(model.initial_results)
    try:
        ensemble_model = AutoTS(forecast_length=10,
                                frequency=FREQUENCY,
                                models_to_validate=MODELS_TO_VALIDATE,
                                no_negatives=NO_NEGATIVES,
                                ensemble='simple',
                                max_generations=0,
                                num_validations=3,
                                validation_method='backward',
                                n_jobs=N_JOBS,
                                verbose=VERBOSE,
                                metric_weighting=METRIC_WEIGHTING,
                                models_mode='default'
                                )

        ensemble_model = ensemble_model.import_template(ens_temp, method='only')
        ensemble_model = ensemble_model.fit(ts.reset_index(), date_col=DATE_COL, value_col=VALUE_COL, id_col=None)

        best_ensemble_models = ensemble_model.export_template(models='best', n=100, max_per_model_class=None,
                                                              include_results=True)
        best_ensemble_models = best_ensemble_models[best_ensemble_models.Ensemble == 1]
        best_models = pd.concat([best_simple_models, best_ensemble_models]) \
            .sort_values('Score', ignore_index=True)

        return ensemble_model, best_models

    except:
        return model, best_simple_models
