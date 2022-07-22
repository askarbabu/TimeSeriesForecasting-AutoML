inputTableLocation = 'revenue_input_v3.csv'
primary_key = ['CompanyID']
COMPANY_LIST = ['HCW', 'Optiqua', 'Micropharma']
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
autots_hyperparameter_tuning = True
max_generations = 1
num_validations = 0
models_to_validate = 0.20
forecast_period = 48

params = {'validation_points': [4, 6, 8, 10], 'validation_method': ['backward', 'even', 'similarity']}

models_with_custom_interval = ['UnivariateRegression', 'MultivariateRegression', 'ETS',
                               'DatepartRegression', 'SectionalMotif', 'GLM']
