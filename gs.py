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
#test_data = copy.deepcopy(train_data.iloc[150000:])
train_data = train_data.iloc[:50000]

#y_true = test_data['loss']

#ids = test_data['id']

target = train_data['loss']

#drop the unnecessary column id and loss from both train and test set.
train_data.drop(['id','loss'],1,inplace=True)
#test_data.drop(['id','loss'],1,inplace=True)

shift = 200
target = np.log(target+shift)

#merging both the datasets to make single joined dataset
joined = pandas.concat([train_data],ignore_index = True)
del train_data                                       #deleting previous one to save memory.

cat_feature = [n for n in joined.columns if n.startswith('cat')]  #list of all the features containing categorical values

#factorizing them
for column in cat_feature:
    joined[column] = pandas.factorize(joined[column].values, sort=True)[0]
        
del cat_feature

#dividing the training data between training and testing set
train_data = joined.iloc[:50000,:]
#test_data = joined.iloc[150000:,:]

def eval_loss(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

d_train_full = xgb.DMatrix(train_data, label=target)
#d_test = xgb.DMatrix(test_data)

xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=3000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

clf = xgb1.fit(train_data[:],target[:])

from sklearn.grid_search import GridSearchCV

param_test1 = {
 'max_depth':[10,12,14],
 'min_child_weight':[1,3,5,8,10]
}
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=8, cv=5,verbose=2)
gsearch1.fit(train_data[:],target[:])
print (gsearch1.grid_scores_)
print (gsearch1.best_params_)
print (gsearch1.best_score_)
