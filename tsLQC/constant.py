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


std_dev_estimation_len = 4
confidence_interval = 0.90
z_value = 1.64


params = {'validation_points': [4, 6, 8, 10], 'validation_method': ['backward', 'even']}
hp_tuning_models_to_validate = 0.35
hp_tuning_max_generations = 5
hp_tuning_num_validations = 3
hp_tuning_model_list = ['ETS']


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
