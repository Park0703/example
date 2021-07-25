# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 19:03:29 2021

@author: whdrb

데이터전처리 새롭게 해보기.
 - One-hot incoding 도입.
 - 전용면적 데이터 추가.(5의배수, 15~100)
    XGB : 127.9364564286
    RF  : 128.8502040816
    CB  : 124.4633965941
    
 - feature selection
  1) SelectKBest와 feature importance 교집합 추출 (15개 피처)
    XGB : 134.25
    RF  : 132.77
    CB  : 131.47

  2) SelectKBest만 고려 (feature importance은 연속형 피처에만 효과적) (20개 피처)
    XGB : 133.36
    RF  : 129.47
    CB  : 127.48
    
  3) SelectKBest만 고려 (feature importance은 연속형 피처에만 효과적) (피처25개 )
    XGB : 126.47
    RF  : 129.86
    CB  : 121.75

  4) SelectKBest만 고려 (feature importance은 연속형 피처에만 효과적) (28개 피처)
    XGB : 
    RF  : 
    CB  : 129.87
    
  5) SelectKBest 25 + 정규화
    XGB : 
    RF  : 137.29
    CB  : 143.40
    
  6) corr 25개 피처 추출
    XGB : 128.81
    RF  : 128.39
    CB  : 122.66
"""


# Standard library imports
import os 
import glob
import pandas as pd
import numpy as np
import itertools

# Third party imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error         # 평균제곱오차
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as LGB
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.svm import SVR

from tqdm import tqdm
import plotly 
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import matplotlib
from matplotlib import font_manager, rc
import shap

# Dacon plotly 그림 업로드 
pd.options.plotting.backend = 'plotly'
## plotly.io를 import 한 후 renderers 기본값을 꼭 "notebook_connected" 로 설정해주시기 바랍니다.
import plotly.io as pio
pio.renderers.default = "notebook_connected"

#한글폰트 사용하기
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
print(font_name) #맑은고딕
rc('font',family=font_name) #현재 폰트 변경 설정.
#마이너스 숫자 표현하기
pd.set_option('display.max_columns', None)
matplotlib.rcParams['axes.unicode_minus'] = False

#데이터 불러오기
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')

'''
   데이터 전처리
     1. 결측값 및 이상값(-) 처리하기.
     2. 
'''

##############컬럼명 1차변경
# '도보 10분거리 내 지하철역 수(환승노선 수 반영)' >> 지하철
# '도보 10분거리 내 버스정류장 수' >> 버스
train.columns=['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
               '자격유형', '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '등록차량수']
test.columns=['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
              '자격유형', '임대보증금', '임대료', '지하철', '버스', '단지내주차면수']





'''''''''''''''''''''''''''''''''결측처리'''''''''''''''''''''''''''''''''''''''
############# 임대보증금, 임대료 컬럼의 NaN값과 '-'값과 결측값 ####################
#train
####### 임대보증금 결측처리.
#임대보증금 - >> 0으로
idx=train[train['임대보증금']=='-'].index
train.loc[idx,'임대보증금']='0'

#임대보증금 NAN값에 0 넣기.
idx_na=train[train['임대보증금'].isna()].index
train.loc[idx_na,'임대보증금']=0

#임대보증금을 모두 숫자로
train['임대보증금']=pd.to_numeric(train['임대보증금'])


####### 임대료 결측처리
# 임대료 '-' >> 모두 0으로
idx=train[train['임대료']=='-'].index
train.loc[idx,'임대료']='0'

#결측값에 0넣기.
idx_na=train[train['임대료'].isna()].index
train.loc[idx_na,'임대료']=0

#임대료 자료형을 모두 숫자로.
train['임대료']=pd.to_numeric(train['임대료'])

#test
####### 임대보증금 결측처리.
#임대보증금 - >> 0으로
idx=test[test['임대보증금']=='-'].index
test.loc[idx,'임대보증금']='0'

#임대보증금 0값에 평균 넣기.
idx_na=test[test['임대보증금'].isna()].index
test.loc[idx_na,'임대보증금']=0

#임대보증금을 모두 숫자로.
test['임대보증금']=pd.to_numeric(test['임대보증금'])


####### 임대료 결측처리.
#임대료 '-' >> 0으로
idx=test[test['임대료']=='-'].index
test.loc[idx,'임대료']='0'
test.loc[idx,'임대료']

#결측값에 0 넣기.
idx_na=test[test['임대료'].isna()].index
test.loc[idx_na,'임대료']=0

#임대료를 모두 숫자로.
test['임대료']=pd.to_numeric(test['임대료'])

######################## 지하철,버스 결측 처리. ################################
#1) 지하철.
# 0로 처리
idx_sub=train[train['지하철'].isna()].index
train.loc[idx_sub,'지하철']=0

#0로 처리
idx_sub=test[test['지하철'].isna()].index
test.loc[idx_sub,'지하철']=0

#2) 버스.
# 0으로 처리 : test에는 버스 결측값이 없어서 아예 버스정류장이 없다고 가정함.
idx_bus=train[train['버스'].isna()].index
train.loc[idx_bus,'버스']=0



######################## 자격유형의 결측값 처리.#################################
# 각 단지코드와 같은 자격유형으로 처리한다.
test[test['자격유형'].isna()].index #196,258
# 2-1 196인덱스의 단지코드를 구한다.
test.loc[196,['단지코드','자격유형']] #C2411
test[test["단지코드"]=='C2411'].loc[:,'자격유형'] #대부분 A 따라서 A로 맞춤.
test.loc[196,'자격유형']='A'

# 2-2 258인덱스의 단지코드를 구한다.
test.loc[258,['단지코드','자격유형']] #C2253
test[test['단지코드']=='C2253'].loc[:,'자격유형'] #임대료가 존재하므로 C로
test.loc[258,'자격유형']='C'

'''
대회측 오류
1. 전용면적별 세대수 합계와 총세대수가 일치하지 않는 경우
2. 동일한 단지에 단지코드가 2개로 부여된 경우  
3. 단지코드 등 기입 실수로 데이터 정제 과정에서 매칭 오류 발생
'''

# 1) 전용면적별 세대수 합계와 총세대수가 일치하지 않는 경우
#해당 행 삭제하기. (train셋만 해당.)
list1=['C1490','C2497','C2620','C1344','C1024','C2470','C1206','C1740','C2405','C1804']

for i in list1:
    idx=train[train['단지코드']==i].index
    train=train.drop(idx)

#2) 동일한 단지에 단지코드가 2개로 부여된 경우
#코드쌍 : ['C2085', 'C1397'], ['C2431', 'C1649'], ['C1036', 'C2675'] 
# 삭제 : 'C1036'(train), 'C1397'(train), 'C2431'(train)
list2=['C1397','C2431','C1036']

for i in list2:
    idx=train[train['단지코드']==i].index
    train=train.drop(idx)

#3) 단지코드 등 기입 실수로 데이터 정제 과정에서 매칭 오류 발생
list3_train=['C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']

for i in list3_train:
    idx=train[train['단지코드']==i].index
    train=train.drop(idx)





'''''''''''''''''''''''''''''''''''''중복행 제거'''''''''''''''''''''''''''''''
train.shape, train.drop_duplicates().shape #((2830, 15), (2517, 15))
test.shape, test.drop_duplicates().shape  #((1022, 14), (949, 14))

train = train.drop_duplicates()
test = test.drop_duplicates()






''''''''''''''''''''''''''''''''''unique() 필터링 :  One-hot-encoding'''''''''''''''
###############################단지 내 유일하지 않는 컬럼 처리 ####################
train.groupby(['단지코드']).nunique(dropna=False)
# 404가 아닌 컬럼들은 모두 하나 이상의 unique() 를 갖고있는 것.
train.groupby(['단지코드']).nunique(dropna=False).sum(axis=0)

#유일한 값을 가진 컬럼만 모아놓기.
unique_cols=['총세대수', '지역', '지하철', '버스', '단지내주차면수', '등록차량수']

#취합을 위해 '단지코드'를 인덱스로.
train_agg = train.set_index('단지코드')[unique_cols].drop_duplicates()
test_agg = test.set_index('단지코드')[[col for col in unique_cols if col!='등록차량수']].drop_duplicates()

#원-핫-인코딩 도입.
def reshape_cat_features(data, cast_col, value_col):
    res = data.drop_duplicates(['단지코드', cast_col]).assign(counter=1).pivot(index='단지코드', columns=cast_col, values=value_col).fillna(0)
    res.columns.name = None
    res = res.rename(columns={col:cast_col+'_'+col for col in res.columns})
    return res

#############################임대건물구분 처리
reshape_cat_features(data=train, cast_col='임대건물구분', value_col='counter')
reshape_cat_features(data=test, cast_col='임대건물구분', value_col='counter')

##################################공귭유형 처리.
#공공임대와 국민임대로 나눔
#https://brunch.co.kr/@leeeeesh/91 참고.
#공공임대5년, 공공뷴양, 공공임대(10년), 공공임대(분납)을 한대로 묶기.
#장기전세, 국민임대를 한대로 묶기.
train.loc[train.공급유형.isin(['공공임대(5년)', '공공분양', '공공임대(10년)', '공공임대(분납)']), '공급유형'] = '공공임대(5년/10년/분납/분양)'
test.loc[test.공급유형.isin(['공공임대(5년)', '공공분양', '공공임대(10년)', '공공임대(분납)']), '공급유형'] = '공공임대(5년/10년/분납/분양)'
train.loc[train.공급유형.isin(['장기전세', '국민임대']), '공급유형'] = '국민임대/장기전세'
test.loc[test.공급유형.isin(['장기전세', '국민임대']), '공급유형'] = '국민임대/장기전세'

#공급유형 처리
reshape_cat_features(data=train, cast_col='공급유형', value_col='counter')
reshape_cat_features(data=test, cast_col='공급유형', value_col='counter')


#################################자격유형 처리
#공급유형별로 자격유형을 나눠 처리
#https://www.myhome.go.kr/hws/portal/cont/selectContRentalView.do#guide=RH101 참고.
train.loc[train.자격유형.isin(['J', 'L', 'K', 'N', 'M', 'O']), '자격유형'] = '행복주택_공급대상'
test.loc[test.자격유형.isin(['J', 'L', 'K', 'N', 'M', 'O']), '자격유형'] = '행복주택_공급대상'

train.loc[train.자격유형.isin(['H', 'B', 'E', 'G']), '자격유형'] = '국민임대/장기전세_공급대상'
test.loc[test.자격유형.isin(['H', 'B', 'E', 'G']), '자격유형'] = '국민임대/장기전세_공급대상'

train.loc[train.자격유형.isin(['C', 'I', 'F']), '자격유형'] = '영구임대_공급대상'
test.loc[test.자격유형.isin(['C', 'I', 'F']), '자격유형'] = '영구임대_공급대상'

reshape_cat_features(data=train, cast_col='자격유형', value_col='counter')
reshape_cat_features(data=test, cast_col='자격유형', value_col='counter')

#종합하기
train_agg = pd.concat([train_agg,
                       reshape_cat_features(data=train, cast_col='임대건물구분', value_col='counter'),
                       reshape_cat_features(data=train, cast_col='공급유형', value_col='counter'),
                       reshape_cat_features(data=train, cast_col='자격유형', value_col='counter')], axis=1)

test_agg = pd.concat([test_agg,
                       reshape_cat_features(data=test, cast_col='임대건물구분', value_col='counter'),
                       reshape_cat_features(data=test, cast_col='공급유형', value_col='counter'),
                       reshape_cat_features(data=test, cast_col='자격유형', value_col='counter')], axis=1)

#컬럼확인하기
train_agg.shape, test_agg.shape #((404, 19), (150, 18))





'''''''''''''''''''''''''''''''전용면적 데이터 추합'''''''''''''''''''''''''''''''
#################전용면적은 5의 배수로, 범위는 15~100까지.
#5의 배수로 만들기
train['전용면적']=train['전용면적']//5*5
test['전용면적']=test['전용면적']//5*5

#상한선 규명.
idx=train[train['전용면적']>100].index
train.loc[idx,'전용면적']=100
idx=test[test['전용면적']>100].index
test.loc[idx,'전용면적']=100

#하한선 규명.
idx=train[train['전용면적']<15].index
train.loc[idx,'전용면적']=15
idx=test[test['전용면적']<15].index
test.loc[idx,'전용면적']=15

######################### train, test 데이터를 1차원으로 규합.
#컬럼=변수. 모든컬럼.
columns=['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '자격유형',
               '임대보증금', '임대료', '지하철', '버스', '단지내주차면수']
target='등록차량수'
area_columns=[]
#f'면적_{area}' : 전용면적을 과 "면적_60.0"같은 형태로 저장.
for area in train['전용면적'].unique():
    area_columns.append(f'면적_{area}')
print(area_columns)
new_train = pd.DataFrame()
new_test = pd.DataFrame()

from tqdm import tqdm
# 단지별로 취합하기.
for i,code in tqdm(enumerate(train['단지코드'].unique())):
    temp=train[train['단지코드']==code] #해당 단지와 일치하는 모든 정보를 temp에 저장.
    temp.index=range(temp.shape[0]) #temp.index의 인덱스 카운트(행의 수)
    for col in columns:
        new_train.loc[i,col]=temp.loc[0,col] #각 단지(i)의 train정보(temp)들 중 
                                             #'columns'에 해당하는 정보들만 new_train에 저장.
    for col in area_columns: #5의 배수로 나눈 면적컬럼들.
        area=float(col.split('_')[-1]) #각 면적컬럼값과 일치하는  '전용면적별세대수'의 합을 new_train에 저장.
        new_train.loc[i,col]=temp[temp['전용면적']==area]['전용면적별세대수'].sum()

    new_train.loc[i,'등록차량수']=temp.loc[0,'등록차량수']

for i, code in tqdm(enumerate(test['단지코드'].unique())):
    temp = test[test['단지코드']==code]
    temp.index = range(temp.shape[0])
    for col in columns:
        new_test.loc[i, col] = temp.loc[0, col]
    
    for col in area_columns:
        area = float(col.split('_')[-1])
        new_test.loc[i, col] = temp[temp['전용면적']==area]['전용면적별세대수'].sum()

##########################면적별 세대수 데이터를 단지코드로 규합.
col_area=['단지코드','면적_30.0', '면적_35.0', '면적_45.0', '면적_50.0', '면적_40.0', '면적_55.0', '면적_25.0', '면적_70.0',
          '면적_15.0', '면적_20.0', '면적_100.0', '면적_60.0', '면적_75.0', '면적_80.0', '면적_65.0']
area_train=new_train[col_area]
area_test=new_test[col_area]

##########################train_agg, test_agg 와 결합하기 (학습, 테스트용)
train_agg=pd.merge(train_agg,area_train, on='단지코드')
test_agg=pd.merge(test_agg,area_test, on='단지코드')

#컬럼확인하기
train_agg.shape, test_agg.shape #((404, 35), (150, 34))






'''''''''''''''''''''''''''''''''Feature Selection'''''''''''''''''''''''''''''''''
########################### 1. Univariate Selection
target_col = '등록차량수'
X=train_agg.drop(columns=[target_col,'단지코드','지역'])
Y=train_agg[target_col]


from sklearn.feature_selection import SelectKBest,chi2
select=SelectKBest(score_func=chi2, k=25)
fit=select.fit(X,Y)
feature=fit.transform(X)

#feature 컬럼명 찾기.
column_names = [column[0]  for column in zip(X.columns,select.get_support()) if column[1]]
column_names = X.columns[select.get_support()]

#컬럼명 지정
feature=pd.DataFrame(feature)
column_names=column_names.tolist()
feature.columns=column_names
len(feature.columns)

feature.columns
'''
'총세대수', '지하철', '버스', '단지내주차면수', '임대건물구분_상가', '공급유형_공공임대(50년)',
       '공급유형_공공임대(5년/10년/분납/분양)', '공급유형_영구임대', '공급유형_임대상가', '자격유형_D',
       '자격유형_영구임대_공급대상', '면적_30.0', '면적_35.0', '면적_45.0', '면적_50.0', '면적_40.0',
       '면적_55.0', '면적_25.0', '면적_70.0', '면적_15.0', '면적_20.0', '면적_100.0',
       '면적_60.0', '면적_80.0', '면적_65.0'
'''


new_col=['단지코드','지역','총세대수', '지하철', '버스', '단지내주차면수', '임대건물구분_상가', '공급유형_공공임대(50년)',
       '공급유형_공공임대(5년/10년/분납/분양)', '공급유형_영구임대', '공급유형_임대상가', '자격유형_D',
       '자격유형_영구임대_공급대상', '면적_30.0', '면적_35.0', '면적_45.0', '면적_50.0', '면적_40.0',
       '면적_55.0', '면적_25.0', '면적_70.0', '면적_15.0', '면적_20.0', '면적_100.0',
       '면적_60.0', '면적_80.0', '면적_65.0','등록차량수']

len(new_col)

train_agg=train_agg[new_col]
test_agg=test_agg[new_col[0:27]]

########################### 2. Correlation Selection
corr = train_agg.drop(['단지코드','지역'],1,).corr()["등록차량수"].abs().sort_values(ascending=False)
corr
'''
등록차량수                      1.000000
단지내주차면수                    0.852745
총세대수                       0.603807
공급유형_공공임대(5년/10년/분납/분양)    0.394441
면적_55.0                    0.390710
면적_45.0                    0.390211
면적_70.0                    0.350513
임대건물구분_상가                  0.311484
공급유형_임대상가                  0.311484
면적_50.0                    0.281322
자격유형_D                     0.265910
면적_80.0                    0.250215
면적_100.0                   0.247175
자격유형_A                     0.228166
면적_35.0                    0.221946
면적_65.0                    0.194124
공급유형_영구임대                  0.192740
자격유형_영구임대_공급대상             0.187965
공급유형_행복주택                  0.178772
자격유형_행복주택_공급대상             0.178772
자격유형_국민임대/장기전세_공급대상        0.120495
공급유형_공공임대(50년)             0.109185
공급유형_국민임대/장기전세             0.106440
면적_60.0                    0.103659
버스                         0.102995
면적_75.0                    0.097360
면적_30.0                    0.081763
면적_15.0                    0.078794
면적_20.0                    0.072531
면적_25.0                    0.046686
지하철                        0.008872
면적_40.0                    0.007468
'''
cor_col=corr.index[1:26]
cor_col=['단지코드','지역','단지내주차면수', '총세대수', '공급유형_공공임대(5년/10년/분납/분양)',
         '면적_55.0', '면적_45.0','면적_70.0', '임대건물구분_상가', '공급유형_임대상가', '면적_50.0',
         '자격유형_D', '면적_80.0', '면적_100.0', '자격유형_A', '면적_35.0', '면적_65.0', '공급유형_영구임대',
         '자격유형_영구임대_공급대상', '공급유형_행복주택', '자격유형_행복주택_공급대상', 
         '자격유형_국민임대/장기전세_공급대상','공급유형_공공임대(50년)', '공급유형_국민임대/장기전세',
         '면적_60.0', '버스', '면적_75.0', '등록차량수']

len(cor_col)

train_agg=train_agg[cor_col]
test_agg=test_agg[cor_col[0:27]]



''''''''''''''''''''''''''''''''''학습 및 모델링.''''''''''''''''''''''''''''''''''
########################## 1. CB
cat_features = ['지역']
target_col = '등록차량수'

from catboost import CatBoostRegressor
catb = CatBoostRegressor(
         cat_features=cat_features,
         loss_function='MAE',
         n_estimators=500, 
         learning_rate=0.05, 
         random_state=42
    )
    
catb.fit(train_agg.drop(columns=[target_col,'단지코드']), train_agg[target_col], verbose=100)

submission['num'] = catb.predict(test_agg.drop('단지코드',axis=1))

submission.to_csv('NEW_CB_corr25.csv', index=False)

########################### 2. XGB
target_col = '등록차량수'

model_XGB=XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                       colsample_bytree=1, max_depth=7)

model_XGB.fit(train_agg.drop(columns=[target_col,'단지코드','지역']), train_agg[target_col])

#test데이터 수정
submission['num']=model_XGB.predict(test_agg.drop(columns=['단지코드','지역']))

submission.to_csv('NEW_XGB_corr25.csv', index=False)


########################## 3. RF
target_col = '등록차량수'

x_train = train_agg.drop(columns=[target_col,'단지코드','지역']) #모든행,  두번째열~마지막 이전 열
y_train = train_agg[target_col]    #마지막 열(예측하고자 하는 컬럼)만 저장.
x_test = test_agg.drop(columns=['단지코드','지역'])      #예측이 맞는 지 검증.

#n_jobs=-1 : 이 컴퓨터의 모든 코어 사용
#random_state=1 : 학습 시, 레코드를 추출할 때 고정된 하나의 레코드들만 추출.
#random_state=42 : 학습 시, 레코드를 추출할 때, 42개의 정해진 레코드셋을 추출.
model = RandomForestRegressor(n_jobs=-1, random_state=42)

model.fit(x_train, y_train) #학습시키는 명령어.

pred = model.predict(x_test)   #42개 레코드셋 추출.

submission['num'] = pred     #예측된 x_test 값을 'num'을 저장.

submission.to_csv('NEW_RF_corr25.csv', index=False)



































































