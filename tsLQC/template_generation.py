import itertools
import json
import pandas as pd

f = open('model_params.json', "r")
params_dict = json.loads(f.read())


def generatetemplate(modelname, params):

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


def template_generation():
    temp_df = pd.DataFrame()

    for i in params_dict.keys():
        temp_df = pd.concat([temp_df, generatetemplate(i, params_dict[i])])

    df = temp_df.reset_index().rename(columns={'index': 'ID'})

    return df
