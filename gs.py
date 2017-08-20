# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

#importing the  necessary modules
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

#dividing the training data between training and testing set
train_data = joined.iloc[:150000,:]
test_data = joined.iloc[150000:,:]

def eval_loss(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
d_train_full = xgb.DMatrix(train_data, label=target)
d_test = xgb.DMatrix(test_data)
def modelfit(alg,train_data,target,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_data, label=target)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            feval=eval_loss, early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train_data, target,eval_metric=eval_loss)

xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, train_data[:100], target[:100])

clf = xgb1.fit(train_data[:100],target[:100])

from sklearn.grid_search import GridSearchCV

param_test1 = {
 'max_depth':[10,11,12,13,14,15],
 'min_child_weight':[1,3,5,8,10]
}
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=1,iid=False, cv=5)
gsearch1.fit(train_data[:],target[:])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
