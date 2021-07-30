# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 18:35:26 2021

@author: whdrb
"""
'''
프로그램 이름.
1. baseline : ㄹㅇ 그냥 베이스라인 130
2. data_preprocessing : 모든 데이터를 숫자전처리로 변수반영. 114
3. ALL_data_recover_data_error : 모든 오류처리한 파일. 120
4. ALL_data_recover_data_error2 : 2,3번 오류만 처리한 파일. 118
5. ADD_AGE : age_gender.csv 모든 데이터와 결합.(오류반영X) 118
6. ADD_AGE_recover1 : age_gender.csv 모든 데이터와 결합.(오류반영O)
'''



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error         # 평균제곱오차
from sklearn.ensemble import RandomForestRegressor
from matplotlib import font_manager, rc
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
print(font_name) #맑은고딕
rc('font',family=font_name) #현재 폰트 변경 설정.

#데이터 불러오기
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
age=pd.read_csv('age_gender_info.csv')
submission=pd.read_csv('sample_submission.csv')

#기존 컬럼명 변경.
train.columns=['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
       '신분', '임대보증금', '임대료', '지하철','버스', '단지내주차면수', '등록차량수']
test.columns=['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
       '신분', '임대보증금', '임대료', '지하철','버스', '단지내주차면수']



'''
   age dataset 결합시키기.
    방법 1. age의 모든 데이터 살리기. (V)
    방법 2. age의 10대, 90대, 100대 데이터 날리기 ()
    방법 3. age의 20대 밑으로, 80대 위로 값 결합시키기. ()
'''
age=age.set_index('지역')
for i in age.index:
    idx=train[train['지역']==i].index
    for j in age.columns:
        train.loc[idx,j]=age.loc[i,j]
        
for i in age.index:
    idx=test[test['지역']==i].index
    for j in age.columns:
        test.loc[idx,j]=age.loc[i,j]

#데이터의 컬럼명 변경하기.
train=train[['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
               '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '10대미만(여자)',
               '10대미만(남자)', '10대(여자)', '10대(남자)', '20대(여자)', '20대(남자)', '30대(여자)',
               '30대(남자)', '40대(여자)', '40대(남자)', '50대(여자)', '50대(남자)', '60대(여자)',
               '60대(남자)', '70대(여자)', '70대(남자)', '80대(여자)', '80대(남자)', '90대(여자)',
               '90대(남자)', '100대(여자)', '100대(남자)', '등록차량수']]
test=test[['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
       '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '10대미만(여자)', '10대미만(남자)',
       '10대(여자)', '10대(남자)', '20대(여자)', '20대(남자)', '30대(여자)', '30대(남자)',
       '40대(여자)', '40대(남자)', '50대(여자)', '50대(남자)', '60대(여자)', '60대(남자)',
       '70대(여자)', '70대(남자)', '80대(여자)', '80대(남자)', '90대(여자)', '90대(남자)',
       '100대(여자)', '100대(남자)']]


test.columns
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

#검증
for i in list1:
    print(train[train['단지코드']==i])

    
#2) 동일한 단지에 단지코드가 2개로 부여된 경우
#코드쌍 : ['C2085', 'C1397'], ['C2431', 'C1649'], ['C1036', 'C2675'] 
# 삭제 : 'C1036'(train), 'C1397'(train), 'C2431'(train)

list2=['C1397','C2431','C1036']
# train셋 삭제
for i in list2:
    idx=train[train['단지코드']==i].index
    train=train.drop(idx)

#검증
for i in list2:
    print(train[train['단지코드']==i])


#3) 단지코드 등 기입 실수로 데이터 정제 과정에서 매칭 오류 발생

list3_train=['C1095', 'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988']

for i in list3_train:
    idx=train[train['단지코드']==i].index
    train=train.drop(idx)

#검증
for i in list3_train:
    print(train[train['단지코드']==i])



'''
데이터 전처리
  고려할 변수
  - 문자열 => 번호 가공이 필요한 변수
    > 단지코드                       번호 (0~422) 423개 범주
    > 임대구분.(아파트, 상가)         번호 (0,1)   2개 범주
    > 지역                           번호 (0~15)  16개 범주
    > 공급유형                       번호 (0~9)   10개 범주
    > 신분                           번호 (0~15)  16개 범주
    
  - 범위를 지정하여 각각의 변수로 저장할 필요가 있는 변수.
    단지코드별로 묶기 위해서 범위를 지정하여 각각의 변수로 치환할 필요가 있음.
    > 전용면적 + 전용면적별 세대수(sum)
    > 임대보증금. 단위별로 나누어 해당단위의 세대수를 변수로 저장.
    > 임대료.     단위별로 나누어 해당단위의 세대수를 변수로 저장.
    
  - 그대로 사용할 변수
    > 총세대수
    > 지하철
    > 공가수
    > 버스
    > 단지내주차면수

  통합할 데이터
    > age DataFrame : 지역별, 연령별, 성별별 인구의 비율이 저장된 데이터프레임.

>>등록차량 수 계산.
'''
##### 1. 결측치 처리.
#test.isna().sum()
#train.isna().sum()

#train.info()
############# 임대보증금, 임대료 컬럼의 NaN값과 '-'갑 처리하기.
#train
####### 임대보증금 결측처리.
#임대보증금 - >> 0으로
idx=train[train['임대보증금']=='-'].index
train.loc[idx,'임대보증금']='0'
train.loc[idx,'임대보증금']

#임대보증금을 모두 숫자로
train['임대보증금']=pd.to_numeric(train['임대보증금'])
#NAN값에 넣을 평균값 준비.
avg=train['임대보증금'].mean()
#임대보증금 NAN값에 평균 넣기.
idx_na=train[train['임대보증금'].isna()].index
train.loc[idx_na,'임대보증금']=avg


####### 임대료 결측처리
# 임대료 '-' >> 모두 0으로
idx=train[train['임대료']=='-'].index
train.loc[idx,'임대료']='0'
train.loc[idx,'임대료']

#임대료 자료형을 모두 숫자로.
train['임대료']=pd.to_numeric(train['임대료'])
#NAN갑에 넣을 평균준비.
avg=train['임대료'].mean()
#결측값에 평균값넣기.
idx_na=train[train['임대료'].isna()].index
train.loc[idx_na,'임대료']=avg


############# 임대보증금, 임대료 컬럼의 NaN값과 '-'갑 처리하기.
#test
####### 임대보증금 결측처리.
#임대보증금 - >> 0으로
idx=test[test['임대보증금']=='-'].index
test.loc[idx,'임대보증금']='0'
test.loc[idx,'임대보증금']

#임대보증금을 모두 숫자로.
test['임대보증금']=pd.to_numeric(test['임대보증금'])
#NAN값에 넣을 평균값 준비.
avg=test['임대보증금'].mean()
#임대보증금 NAN값에 평균 넣기.
idx_na=test[test['임대보증금'].isna()].index
test.loc[idx_na,'임대보증금']=avg

####### 임대료 결측처리.
#임대료 '-' >> 0으로
idx=test[test['임대료']=='-'].index
test.loc[idx,'임대료']='0'
test.loc[idx,'임대료']

#임대료를 모두 숫자로.
test['임대료']=pd.to_numeric(test['임대료'])
#결측값에 들어갈 test의 '임대료'의 평균구하기
avg=test['임대료'].mean()
#결측값에 평균 넣기.
idx_na=test[test['임대료'].isna()].index
test.loc[idx_na,'임대료']=avg

#train.isna().sum()
#test.isna().sum()
#train.info()
#test.info()

####### 지하철 결측 처리.
#1) train
# -1로 처리
idx_sub=train[train['지하철'].isna()].index
train.loc[idx_sub,'지하철']=-1

#2) test
#-1로 처리
idx_sub=test[test['지하철'].isna()].index
test.loc[idx_sub,'지하철']=-1

################## 나머지 결측.
# 1. train의 버스
# 0으로 처리 : test에는 버스 결측값이 없어서 아예 버스정류장이 없다고 가정함.
idx_bus=train[train['버스'].isna()].index
train.loc[idx_bus,'버스']=0

# 2. test의 신분
# 각 단지코드와 같은 신분으로 처리한다.
test[test['신분'].isna()].index #196,258
# 2-1 196인덱스의 단지코드를 구한다.
test.loc[196,['단지코드','신분']] #C2411
test[test["단지코드"]=='C2411'].loc[:,'신분'] #대부분 A 따라서 A로 맞춤.
test.loc[196,'신분']='A'

# 2-2 258인덱스의 단지코드를 구한다.
test.loc[258,['단지코드','신분']] #C2253
test[test['단지코드']=='C2253'].loc[:,'신분'] #대부분 D 따라서 D로 맞춤.
test.loc[258,'신분']='D'

'''
유의미한 문자인덱스를 숫자인덱스로 변환하기.
    > 임대구분.(아파트, 상가)         번호 (0,1)   2개 범주
    > 지역                           번호 (0~15)  16개 범주
    > 공급유형                       번호 (0~9)   10개 범주
    > 신분                           번호 (0~15)  16개 범주
'''
# 1. 지역들을 번호로 마킹하기.
#각 지역명을 저장할 디렉토리.
local_map={}
type(local_map)

for i,loc in enumerate(train['지역'].unique()):
    #i : 인덱스
    #loc : 지역명.
    local_map[loc]=i
print(local_map)
train['지역']=train['지역'].map(local_map)
test['지역']=test['지역'].map(local_map)
print(train['지역'].unique())
print(test['지역'].unique())

# 2. 신분들을 번호로 마킹하기.
#각 신분등급을 저장할 디렉토리 형성.
grade_map={}
type(grade_map)
train['신분'].unique()

for i,loc in enumerate(train['신분'].unique()):
    grade_map[loc]=i
train['신분']=train['신분'].map(grade_map)
test['신분']=test['신분'].map(grade_map)
print(train['신분'].unique())
print(test['신분'].unique())

# 3. 공급유형을 번호로 마킹하기.
#공급유형을 저장할 디렉토리 생성.
gong_map={}
train['공급유형'].unique()

for i,loc in enumerate(train['공급유형'].unique()):
    gong_map[loc]=i
train['공급유형']=train['공급유형'].map(gong_map)
test['공급유형']=test['공급유형'].map(gong_map)
train['공급유형'].unique()
test['공급유형'].unique()

# 4. 임대건물구분
#임대건물구분별로 저장할 디랙토리 형성
imdae_map={}
train['임대건물구분'].unique()

for i,loc in enumerate(train['임대건물구분'].unique()):
    gong_map[loc]=i
train['임대건물구분']=train['임대건물구분'].map(gong_map)
test['임대건물구분']=test['임대건물구분'].map(gong_map)
train['임대건물구분'].unique()
test['임대건물구분'].unique()

#단지코드를 제외한 모든 변수 숫자화(int,float화 완성.)
train.info()

'''
전용면적 만들기 : 
    실험값 : 5의 배수, 3의 배수 등등..
    상/하한 적용 : 

'''
#실험 1. 전용면적 5배.
train['전용면적']=train['전용면적']//5*5
test['전용면적']=test['전용면적']//5*5
train['전용면적'].unique()
print(test['전용면적'].unique())


#상/하한선 정하기.
train.groupby('전용면적')['전용면적'].count()
'''
전용면적
10.0       8
15.0      74
20.0     156
25.0     324
30.0     263
35.0     667
40.0      41
45.0     644
50.0     388
55.0     203
60.0       8
65.0       9
70.0      48
75.0      11
80.0      80
105.0      1
125.0      5
135.0      4
240.0      6
245.0      2
315.0      1
400.0      3
405.0      1
580.0      5
Name: 전용면적, dtype: int64
'''
#상한선 정하기
train[train['전용면적']>100] #28

#하한선 정하기
train[train['전용면적']<15] #8개.

### 상/하한선을 15~100으로 했을 경우, 36(28개, 8개)개의 데이터만을 손해보기때문에
# 합리적이라고 판단.
idx=train[train['전용면적']>100].index
train.loc[idx,'전용면적']=100
idx=test[test['전용면적']>100].index
test.loc[idx,'전용면적']=100

#하한
idx=train[train['전용면적']<15].index
train.loc[idx,'전용면적']=15
idx=test[test['전용면적']<15].index
test.loc[idx,'전용면적']=15

print(train['전용면적'].unique())
print(test['전용면적'].unique())


'''
'''
#각 변수간 상관관계 도출하기.
'''
#단지코드 뺴고 숫자로 이루어진 값들 상관관계 구하기.
train.columns
test.columns
col_train=['총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',\
       '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '등록차량수']
col_test=['총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분', \
       '임대보증금', '임대료', '지하철', '버스', '단지내주차면수']
corr_train=train[col_train].corr()
corr_test=test[col_test].corr()

#상관관계 시각화하기
import seaborn as sns
hm=sns.heatmap(corr_train.values,
               cbar=True,
               annot=True,
               square=True,
               fmt='.2f',
               annot_kws={'size':5},
               yticklabels=col_train,
               xticklabels=col_train)


sns.pairplot(train[col_train],kind="reg")
'''


##################################################################################################################
##################################################################################################################
###########################################   전처리 끝 ###########################################################
##################################################################################################################
##################################################################################################################





'''
단지별 1차원으로 취합하기
'''
train.columns
#컬럼=변수. 모든컬럼.
columns=['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
               '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '10대미만(여자)',
               '10대미만(남자)', '10대(여자)', '10대(남자)', '20대(여자)', '20대(남자)', '30대(여자)',
               '30대(남자)', '40대(여자)', '40대(남자)', '50대(여자)', '50대(남자)', '60대(여자)',
               '60대(남자)', '70대(여자)', '70대(남자)', '80대(여자)', '80대(남자)', '90대(여자)',
               '90대(남자)', '100대(여자)', '100대(남자)']
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
        
new_train.info()
new_test.info()

new_train.isna().sum()
new_test.isna().sum()


'''
 학습
   RF 모델링 이용.
   변수값 28개.
   변수 : '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '신분',
         '임대보증금', '임대료', '지하철', '버스', '단지내주차면수'
'''
#### 본격적인 학습 시작.
x_train = new_train.iloc[:, 1:-1] #모든행,  두번째열~마지막 이전 열
y_train = new_train.iloc[:,-1]    #마지막 열(예측하고자 하는 컬럼)만 저장.
x_test = new_test.iloc[:,1:]      #예측이 맞는 지 검증.

#n_jobs=-1 : 이 컴퓨터의 모든 코어 사용
#random_state=1 : 학습 시, 레코드를 추출할 때 고정된 하나의 레코드들만 추출.
#random_state=42 : 학습 시, 레코드를 추출할 때, 42개의 정해진 레코드셋을 추출.
model = RandomForestRegressor(n_jobs=-1, random_state=42)

model.fit(x_train, y_train) #학습시키는 명령어.

pred = model.predict(x_test)   #42개 레코드셋 추출.

submission['num'] = pred     #예측된 x_test 값을 'num'을 저장.

submission.to_csv('ADD_AGE_recover1.csv', index=False)













