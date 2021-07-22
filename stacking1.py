import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error        
from sklearn.ensemble import RandomForestRegressor
from matplotlib import font_manager, rc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from mlxtend.regressor import StackingCVRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression

#### RandomForest
params = { 'n_estimators' : [100, 200, 300],
           'max_depth' : [8, 12, 50, 100],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

# RandomForestRegressor 객체 생성 후 GridSearchCV 수행
rf_reg = RandomForestRegressor\
    (random_state = 0, n_jobs = -1, criterion='mae')
grid_cv = GridSearchCV(rf_reg, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(x_train, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

rf_reg = RandomForestRegressor(n_estimators=300,
                                max_depth = 12,
                                min_samples_leaf=8,
                                min_samples_split=8,
                                random_state=0,
                                n_jobs=-1,
                                criterion='mae')
rf_reg.fit(x_train, y_train) 
rf_pred = rf_reg.predict(x_test)

submission['num'] = rf_pred
submission.to_csv('submission_rf_02.csv', index=False)

#### GradientBoostingRegressor
params = {'n_estimators':[100, 200],
          'max_depth':[4, 6, 8],
          'min_samples_split':[5, 10], 
          'min_samples_leaf':[3, 5, 7, 10]}
gb_reg = GradientBoostingRegressor()
grid_cv = GridSearchCV(gb_reg, param_grid = params, cv = 3)
grid_cv.fit(x_train, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

####
gb_reg = GradientBoostingRegressor(n_estimators=100, 
                                   learning_rate=0.05,
                                   max_depth=8,
                                   min_samples_leaf=5, 
                                   min_samples_split=5,
                                   criterion='mae')
gb_reg.fit(x_train, y_train) 
gb_pred = gb_reg.predict(x_test)

submission['num'] = gb_pred
submission.to_csv('submission_gb_02.csv', index=False)

#### AdaBoost
for i in (10, 20, 30, 40, 100, 150):
  for j in (0.001, 0.01, 0.1):
    model =  AdaBoostRegressor(n_estimators = i, learning_rate = j)
    model.fit(x_train, y_train)
    relation_square = model.score(x_train, y_train)
    print('결정계수 : ', relation_square)
    
ada_reg = AdaBoostRegressor(n_estimators=150, learning_rate=0.1)
ada_reg.fit(x_train, y_train)
ada_pred = ada_reg.predict(x_test)

submission['num'] = ada_pred
submission.to_csv('submission03.csv', index=False)


#### XGBoost
xgb_reg = xgb.XGBRegressor(gamma=0, 
                           learning_rate=0.05, 
                           max_depth=6, 
                           min_child_weight=10, 
                           n_estimators=2200,
                           random_state =7,
                           nthread = 4,
                           colsample_bytree=0.8,
                           colsample_bylevel=0.9)
xgb_reg.fit(x_train, y_train)
xgb_pred = xgb_reg.predict(x_test)

submission['num'] = xgb_pred
submission.to_csv('submission_xgb_02.csv', index=False)

######### stacking

stacked_pred = np.array([rf_pred, gb_pred, ada_pred, xgb_pred])
print(stacked_pred)
stacked_pred = np.transpose(stacked_pred)
print(stacked_pred)

####### rf를 통한 staking
stackrf = RandomForestRegressor(random_state = 0, n_jobs = -1, criterion='mae')
stackrf.fit(stacked_pred, y_train)
stackfinal = stackrf.predict(stacked_pred)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(stackfinal))
# 대회라서 y_test가 없어서 실행이 안됨