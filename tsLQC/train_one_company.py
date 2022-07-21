from autots import AutoTS
from autots_hyperparameter_tuning import hyperparameter_tuning
from user_input import metric_weighting, \
    max_generations, num_validations, models_to_validate

from template_generation import templateGeneration
df = templateGeneration()


def modelling(ts, autots_hyperparameter_tuning = False):

    if autots_hyperparameter_tuning:
        validation_points, validation_method = hyperparameter_tuning(ts, 12)
    else:
        validation_points, validation_method = 4, 'backward'

    model_list = [i for i in df.Model.unique() if (i != 'FBProphet')]
    model = AutoTS(forecast_length=validation_points,
                   frequency='infer',
                   models_to_validate=models_to_validate,
                   no_negatives=True,
                   ensemble='simple',
                   max_generations=max_generations,
                   num_validations=num_validations,
                   validation_method=validation_method,
                   model_list=model_list,
                   n_jobs=7,
                   metric_weighting=metric_weighting,
                   )
    model = model.import_template(df, method='only')
    model = model.fit(ts.reset_index(), date_col='Date', value_col='Value', id_col=None)

    return model
