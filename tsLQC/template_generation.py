
import itertools
import json
import pandas as pd

def generatetemplate(modelname, params):
    
    d={}    
    
    
    paramslst =[]
    my_dict = {}
    for x in list(itertools.product(*params.values())):
        for i in range(len(x)):
            
            my_dict[list(params.keys())[i]] = x[i]

        paramslst.append(json.dumps(my_dict))
    
    d['Model'] = [modelname]*len(paramslst)
    
    d['ModelParameters'] = paramslst
    
    d['TransformationParameters'] = ['''{"fillna": "fake_date", "transformations": {}, "transformation_params":{}}''']*len(paramslst)
    
    d['Ensemble'] = [0]*len(paramslst)
    
    
    return(pd.DataFrame(d))


def templateGeneration():
    
    gen_temp = pd.concat([
                      generatetemplate("ZeroesNaive",         params = {}),

                      generatetemplate("AverageValueNaive",   params = {"method":["Mean","Median","Mode","Midhinge","Weighted_Mean","Exp_Weighted_Mean"]}),

                      generatetemplate("LastValueNaive",      params = {}),

                      generatetemplate("SeasonalNaive",       params = {'method': ['mean', 'median', 'lastvalue'], 'lag_1': [1, 2, 6, 12], 'lag_2': [None, 1, 3, 6,12]}),
    
                      generatetemplate("GLS",                 params = {}),

                      generatetemplate("GLM",                 params = {'family': ['Gaussian','Poisson','Binomial','NegativeBinomial','Tweedie','Gamma'],'constant': [True, False],'regression_type':[None, 'datepart']}),

                      generatetemplate('ARIMA',               params = {'p' : [0,1,2],'d' : [0,1,2], 'q' : [0,1,2], 'regression_type':[None, 'Holiday']}),
                      
                      generatetemplate("ETS",                 params = {"damped_trend" : [True, False], "trend" : [None,'additive','multiplicative'],
                                                                        "seasonal" : [None,'additive','multiplicative'], "seasonal_periods":[6,12]}),
                      
#                      generatetemplate("Greykite",            params = {"holiday" : [True, False], "regression_type" : [None], "growth" : [None, 'linear', 'quadratic', 'sqrt']}),

                      generatetemplate("UnobservedComponents",  params = {}),

                      generatetemplate("DynamicFactor",       params = { 'k_factors': [0, 1, 2, 3, 10],
                                                                                    'factor_order': [0, 1, 2, 3],
                                                                                    'regression_type':[None,'Holiday'],
                                                                                }),
                      generatetemplate("ARDL",                params =  {
                                                                                'lags': [1, 2, 3, 4],
                                                                                'trend': ['n', 'c', 't', 'ct'],
                                                                                'order': [0, 1, 2, 3],
                                                                                'regression_type': [None,'holiday'],
                                                                            }),
                          generatetemplate("FBProphet",           params =  {}),
    
                     generatetemplate("RollingRegression",   params =  {}),
    
#                       generatetemplate("VARMAX",              params =  {}),

#                       generatetemplate("VECM",                params =  {}),


                      generatetemplate("MotifSimulation",     params =  {}),
                      generatetemplate("WindowRegression",    params =  {}),
#                       generatetemplate("VAR",                 params =  {}),
#                       generatetemplate("TFPRegression",       params =  {}),
                      
                      generatetemplate("ComponentAnalysis",   params =  {}),
                      
                      generatetemplate("DatepartRegression",     params =  {}),
                      generatetemplate("UnivariateRegression",     params =  {}),
                      generatetemplate("UnivariateMotif",     params =  {}),
                       generatetemplate("MultivariateMotif",     params =  {}),
                       generatetemplate("NVAR",     params =  {}),
                       generatetemplate("MultivariateRegression",     params =  {}),
                       generatetemplate("SectionalMotif",     params =  {}),
                       generatetemplate("Theta",     params =  {}),
                    #generatetemplate("DynamicFactor",       params =  {}),
                      # generatetemplate("NeuralProphet",     params =  {}),
                       #generatetemplate("DynamicFactorMQ",     params =  {}),
                      # generatetemplate("GluonTS",     params =  {})
    

                     ])

# gen_temp = pd.concat([
#                       generatetemplate('ARIMA',     params = {'p' : [0,1,2],'d' : [0,1,2], 'q' : [0,1,2], 'regression_type':[None, 'Holiday']}),

#                       generatetemplate("ETS",       params = {"damped_trend" : [True, False], "trend" : [None,'additive','multiplicative'],
#                                                             "seasonal" : [None,'additive','multiplicative'], "seasonal_periods":[6,12]})])
    df = gen_temp.reset_index().rename(columns={'index':'ID'})
    
    return df

