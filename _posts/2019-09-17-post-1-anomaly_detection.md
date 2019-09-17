---
title: "Predict"
date: 2019-9-17 15:09:28 -0400
categories: DeepLearning
tags:
- anomaly detection
- LSTM
- RNN
- Predict model
---

# Training of Anomaly detection Model
* 정상 데이터만을 기반으로 LSTM을 활용하여 이상감지 모델을 구현하고 정확도를 확인해본다.
* 100만개의 정상 데이터 2일치 활용 
* 7개 feature 개별 모델 구현

```python
import os 

%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
import requests
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
plt.style.use('bmh')

import keras 
keras.__version__
```

    Using TensorFlow backend.





    '2.2.4'



       
# 1. Training

    * 20대 차량의 데이터로 만든 5개 feature 개별 model
      1.co
      2.co2
      3.pm_1
      4.pm_10
      5.pm25
      
    * feature 별 sequence 3개씩 추가
      30, 45, 60
      > 총 15개의 모델 각각 비교
      
    * Training 차량 번호
       ['72호4284', '72호4285', '72호4256', '72호4300', '72호4283', '72호4294',
       '72호4278', '72호4296', '72호4288', '72호4291', '72호4262', '72호4263',
       '72호4264', '72호4253', '72호4261', '72호4266', '72호4274', '72호4277',
       '72호4282', '72호4281']
       
    * Training 차량 아이디
       ['b49', 'b3B', 'b34', 'b3E', 'b3D', 'b35', 'b46', 'b44', 'b3C', 'b38',
       'b48', 'b30', 'b45', 'b4B', 'b31', 'b39', 'b2F', 'b2E', 'b37', 'b47']
       
       
# 2. test data load & Preprocessing
    
    * 현재까지 학습된 모델
        feature_co2_sequence_30.h5399 kB2일 전
        feature_co2_sequence_45.h5399 kB19시간 전
        feature_co_sequence_30.h5399 kB2일 전
        feature_co_sequence_45.h5399 kB하루 전
        feature_pm10_sequence_30.h5399 kB하루 전
        feature_pm10_sequence_45.h5399 kB11시간 전
        feature_pm25_sequence_30.h5399 kB2일 전
        feature_pm25_sequence_45.h5399 kB11시간 전
        feature_pm_1_sequence_30.h5399 kB2일 전
        feature_pm_1_sequence_45.h5

    * co_30/ co_45 test_SET 준비 


# 3. 비정상 데이터 예측 vs true 값 비교

    * Abnormal 차량 데이터 사용 
      [‘b33’, ‘b4A’, ‘b32’]
      
    * b33 차량의 co 비교

# test data load & Preprocessing


```python
#학습데이터
traindata_path = '/data_0905/df_Carnival_except_0829_0830.csv'
traindata = pd.read_csv(traindata_path)
print('\n Data shape')
print('------------------------')
print(traindata.shape)
print('------------------------')


#검증데이터
testdata_path = '/data_0905/df_Carnival.csv'
testdata = pd.read_csv(testdata_path)
print('\n testdata shape')
print('------------------------')
print(testdata.shape)
print('------------------------')
```

    
     Data shape
    ------------------------
    (1387845, 16)
    ------------------------
    
     testdata shape
    ------------------------
    (2499088, 16)
    ------------------------



```python
testdata.head()
```





```python
'''
Abnormal ‘b33’, ‘b4A’, ‘b32’
normal ‘b36’, ‘b41’, ‘b42’
'''
testdf = testdata
print('testdata.shape',testdata.shape)
print('testdf.shape',testdf.shape)
print("======= testdf['id'].value_counts =======")
testdf['id'].value_counts()
```

    testdata.shape (2499088, 16)
    testdf.shape (2499088, 16)
    ======= testdf['id'].value_counts =======





    b49    85976
    b41    85800
    b38    85776
    b43    85766
    b36    85682
    b48    85618
    b32    85610
    b3B    85598
    b34    85588
    b30    85580
    b3E    85550
    b45    85546
    b3F    85532
    b4B    85526
    b3D    85444
    b31    85444
    b40    85410
    b39    85406
    b35    85404
    b2E    85392
    b2F    85382
    b4A    85368
    b3A    85368
    b46    85306
    b33    85296
    b42    85242
    b37    85048
    b44    84794
    b3C    66772
    b47    38864
    Name: id, dtype: int64




```python
tdf = testdf[testdf.id == 'b33']
tdf
```



```python
feature = 'co'
sequence1 = 30
sequence2 = 45
sequence3 = 60

scaler = MinMaxScaler()
tdf[[feature]] = scaler.fit_transform(tdf[[feature]])
print( feature + '스케일변경')
tdf
```


```python
test_set = tdf[feature].values.tolist()
```

# test co data / b33


```python
# 차트 
plt.figure(figsize=(25,5))
plt.plot(test_set, color='b')
```

<div align="center">
  <img src="./img/output_10_1.png" width="400">  
  <p>test_set timeserise.</p>
</div>


# LSTM Input dimension transfer


```python
seq1 = []
seq2 = []

for i in range(len(test_set)-sequence1):
    seq1.append([test_set[i+j] for j in range(sequence1)])
    
for i in range(len(test_set)-sequence2):
    seq2.append([test_set[i+j] for j in range(sequence2)])

    
test_Set30 = np.asarray(seq1)
test_Set45 = np.asarray(seq2)
print("test_Set30.shape",test_Set30.shape)
print("test_Set45.shape",test_Set45.shape)
```

    test_Set30.shape (85266, 30)
    test_Set45.shape (85251, 45)



```python
test30 = np.reshape(test_Set30, (test_Set30.shape[0], sequence1, 1))
test45 = np.reshape(test_Set45, (test_Set45.shape[0], sequence2, 1))
print("test30.shape",test30.shape)
print("test45.shape",test45.shape)
```

    test30.shape (85266, 30, 1)
    test45.shape (85251, 45, 1)


# Training Model load 


```python
from keras.models import load_model
model1 = load_model("/data/home/1004207/SOCAR/weights/feature_"+str(feature)+"_sequence_"+str(sequence1)+".h5")
model2 = load_model("/data/home/1004207/SOCAR/weights/feature_"+str(feature)+"_sequence_"+str(sequence2)+".h5")
print("sequence1", sequence1)
print("sequence2", sequence2)
```

    sequence1 30
    sequence2 45



```python
test_pred_30 = model1.predict(test30)
test_pred_45 = model2.predict(test45)
```


# 비정상 데이터 예측 vs true 값 비교

    * Abnormal 차량 데이터 사용 
      [‘b33’, ‘b4A’, ‘b32’]
      
    * b33 차량의 co 비교
    * sequence 30 / 45 / 60

# 최종 결과


```python
plt.figure(figsize=(25,5))
plt.plot(test_set, c='b') ## y_true
plt.plot(test_pred_30, c='g')
plt.title("test_Set_feature co and seq_30_id_b33")
plt.legend(['y_ture', 'y_pred'], loc='upper left')
plt.show()

plt.figure(figsize=(25,5))
plt.plot(test_set, c='b') ## y_true
plt.plot(test_pred_45, c='g')
plt.title("test_Set_feature co and seq_45_id_b33")
plt.legend(['y_ture', 'y_pred'], loc='upper left')
plt.show()
```


<div align="center">
  <img src="./img/output_19_0.png" width="400">  
  <p>test_set timeserise.</p>
</div>

<div align="center">
  <img src="./img/output_19_1.png" width="400">  
  <p>test_set timeserise.</p>
</div>
