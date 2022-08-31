primary_key = ['CompanyID']
DATE_COL = 'Date'
VALUE_COL = 'Value'

autots_hyperparameter_tuning = False
max_generations = 15
num_validations = 2
models_to_validate = 0.20
forecast_period = 48
frequency = 'infer'
n_jobs = 'auto'
no_negatives = True
ensemble = None
verbose = -1


metric_weighting = {
            'smape_weighting': 5,
            'mae_weighting': 5,
            'rmse_weighting': 0,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0,
            'spl_weighting': 0,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
                    }
