# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import pandas                                      #to read and manipulate data
import zipfile                                     #to extract data
import numpy as np                                 #for matrix operations
#read will be imported as and when required
#read the train and test zip file
zip_ref = zipfile.ZipFile("train.csv.zip", 'r')    
zip_ref.extractall()                               
zip_ref.close()

train_data = pandas.read_csv("train.csv")

import copy
test_data = copy.deepcopy(train_data.iloc[150000:])
train_data = train_data.iloc[:150000]

y_true = test_data['loss']

ids = test_data['id']

target = train_data['loss']

#drop the unnecessary column id and loss from both train and test set.
train_data.drop(['id','loss'],1,inplace=True)
test_data.drop(['id','loss'],1,inplace=True)

shift = 200
target = np.log(target+shift)

#merging both the datasets to make single joined dataset
joined = pandas.concat([train_data, test_data],ignore_index = True)
del train_data,test_data                                         #deleting previous one to save memory.

cat_feature = [n for n in joined.columns if n.startswith('cat')]  #list of all the features containing categorical values

#factorizing them
for column in cat_feature:
    joined[column] = pandas.factorize(joined[column].values, sort=True)[0]
        
del cat_feature

train_data = joined.iloc[:150000,:]
test_data = joined.iloc[150000:,:]

from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import xgboost as xgb

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):

    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)


    cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-mae-mean'].values[-1]

xgtrain = xgb.DMatrix(train_data, label=target)

num_rounds = 3000
random_state = 2016
num_iter = 25
init_points = 5
params = {'eta': 0.1,'silent': 1,'eval_metric': 'mae','verbose_eval': True,'seed': random_state}

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (5, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })

xgbBO.maximize(init_points=init_points, n_iter=num_iter)


