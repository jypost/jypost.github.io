---
title: "Anomaly detection Predict"
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
traindata_path = '/data/home/1004207/SOCAR/data_0905/df_Carnival_except_0829_0830.csv'
traindata = pd.read_csv(traindata_path)
print('\n Data shape')
print('------------------------')
print(traindata.shape)
print('------------------------')


#검증데이터
testdata_path = '/data/home/1004207/SOCAR/data_0905/df_Carnival.csv'
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>temp</th>
      <th>humidity</th>
      <th>pm_1</th>
      <th>pm25</th>
      <th>pm10</th>
      <th>co</th>
      <th>co2</th>
      <th>speed</th>
      <th>caron</th>
      <th>idx</th>
      <th>car_no</th>
      <th>detect_id</th>
      <th>type</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>b2E</td>
      <td>25.9</td>
      <td>32.7</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.5</td>
      <td>3050</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AA563-2</td>
      <td>72호4277</td>
      <td>G72121004442</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b2E</td>
      <td>25.9</td>
      <td>32.7</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.5</td>
      <td>3050</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AA563-1</td>
      <td>72호4277</td>
      <td>G72121004442</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b2E</td>
      <td>25.9</td>
      <td>32.5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.5</td>
      <td>3040</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AA5A0-2</td>
      <td>72호4277</td>
      <td>G72121004442</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>b2E</td>
      <td>25.9</td>
      <td>32.5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.5</td>
      <td>3040</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AA5A0-1</td>
      <td>72호4277</td>
      <td>G72121004442</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>b2E</td>
      <td>25.9</td>
      <td>32.4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.5</td>
      <td>3040</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AA635-1</td>
      <td>72호4277</td>
      <td>G72121004442</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:11</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>temp</th>
      <th>humidity</th>
      <th>pm_1</th>
      <th>pm25</th>
      <th>pm10</th>
      <th>co</th>
      <th>co2</th>
      <th>speed</th>
      <th>caron</th>
      <th>idx</th>
      <th>car_no</th>
      <th>detect_id</th>
      <th>type</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>427408</th>
      <td>427408</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>17</td>
      <td>17</td>
      <td>21</td>
      <td>1.3</td>
      <td>640</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAC93-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:00</td>
    </tr>
    <tr>
      <th>427409</th>
      <td>427409</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>17</td>
      <td>17</td>
      <td>20</td>
      <td>1.3</td>
      <td>640</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAC93-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:02</td>
    </tr>
    <tr>
      <th>427410</th>
      <td>427410</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>1.3</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB8-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:04</td>
    </tr>
    <tr>
      <th>427411</th>
      <td>427411</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>1.3</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB8-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:06</td>
    </tr>
    <tr>
      <th>427412</th>
      <td>427412</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>19</td>
      <td>1.3</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB9-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:08</td>
    </tr>
    <tr>
      <th>427413</th>
      <td>427413</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>1.3</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB9-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:10</td>
    </tr>
    <tr>
      <th>427414</th>
      <td>427414</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>15</td>
      <td>15</td>
      <td>18</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACBA-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:12</td>
    </tr>
    <tr>
      <th>427415</th>
      <td>427415</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACBA-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:14</td>
    </tr>
    <tr>
      <th>427416</th>
      <td>427416</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDD-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:16</td>
    </tr>
    <tr>
      <th>427417</th>
      <td>427417</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDD-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:18</td>
    </tr>
    <tr>
      <th>427418</th>
      <td>427418</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>13</td>
      <td>13</td>
      <td>16</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDE-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:20</td>
    </tr>
    <tr>
      <th>427419</th>
      <td>427419</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDE-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:22</td>
    </tr>
    <tr>
      <th>427420</th>
      <td>427420</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDF-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:24</td>
    </tr>
    <tr>
      <th>427421</th>
      <td>427421</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDF-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:26</td>
    </tr>
    <tr>
      <th>427422</th>
      <td>427422</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFD-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:28</td>
    </tr>
    <tr>
      <th>427423</th>
      <td>427423</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFD-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:30</td>
    </tr>
    <tr>
      <th>427424</th>
      <td>427424</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFE-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:32</td>
    </tr>
    <tr>
      <th>427425</th>
      <td>427425</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFE-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:34</td>
    </tr>
    <tr>
      <th>427426</th>
      <td>427426</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>15</td>
      <td>15</td>
      <td>18</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFF-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:36</td>
    </tr>
    <tr>
      <th>427427</th>
      <td>427427</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFF-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:38</td>
    </tr>
    <tr>
      <th>427428</th>
      <td>427428</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD22-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:40</td>
    </tr>
    <tr>
      <th>427429</th>
      <td>427429</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>19</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD22-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:42</td>
    </tr>
    <tr>
      <th>427430</th>
      <td>427430</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD23-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:44</td>
    </tr>
    <tr>
      <th>427431</th>
      <td>427431</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>1.3</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD23-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:46</td>
    </tr>
    <tr>
      <th>427432</th>
      <td>427432</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
      <td>1.3</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3E-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:48</td>
    </tr>
    <tr>
      <th>427433</th>
      <td>427433</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3E-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:50</td>
    </tr>
    <tr>
      <th>427434</th>
      <td>427434</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3F-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:52</td>
    </tr>
    <tr>
      <th>427435</th>
      <td>427435</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3F-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:54</td>
    </tr>
    <tr>
      <th>427436</th>
      <td>427436</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>1.3</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD40-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:56</td>
    </tr>
    <tr>
      <th>427437</th>
      <td>427437</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>1.3</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD40-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:58</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>512674</th>
      <td>512674</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D3F-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:58:59</td>
    </tr>
    <tr>
      <th>512675</th>
      <td>512675</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D3F-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:01</td>
    </tr>
    <tr>
      <th>512676</th>
      <td>512676</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4E-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:03</td>
    </tr>
    <tr>
      <th>512677</th>
      <td>512677</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4E-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:05</td>
    </tr>
    <tr>
      <th>512678</th>
      <td>512678</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4F-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:07</td>
    </tr>
    <tr>
      <th>512679</th>
      <td>512679</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4F-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:09</td>
    </tr>
    <tr>
      <th>512680</th>
      <td>512680</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D50-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:11</td>
    </tr>
    <tr>
      <th>512681</th>
      <td>512681</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D50-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:13</td>
    </tr>
    <tr>
      <th>512682</th>
      <td>512682</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6A-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:15</td>
    </tr>
    <tr>
      <th>512683</th>
      <td>512683</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6A-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:17</td>
    </tr>
    <tr>
      <th>512684</th>
      <td>512684</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6B-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:19</td>
    </tr>
    <tr>
      <th>512685</th>
      <td>512685</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6B-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:21</td>
    </tr>
    <tr>
      <th>512686</th>
      <td>512686</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6C-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:23</td>
    </tr>
    <tr>
      <th>512687</th>
      <td>512687</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6C-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:25</td>
    </tr>
    <tr>
      <th>512688</th>
      <td>512688</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D7B-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:27</td>
    </tr>
    <tr>
      <th>512689</th>
      <td>512689</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D7B-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:29</td>
    </tr>
    <tr>
      <th>512690</th>
      <td>512690</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634498-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:31</td>
    </tr>
    <tr>
      <th>512691</th>
      <td>512691</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634498-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:33</td>
    </tr>
    <tr>
      <th>512692</th>
      <td>512692</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634499-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:35</td>
    </tr>
    <tr>
      <th>512693</th>
      <td>512693</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634499-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:37</td>
    </tr>
    <tr>
      <th>512694</th>
      <td>512694</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b463449A-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:40</td>
    </tr>
    <tr>
      <th>512695</th>
      <td>512695</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b463449A-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:42</td>
    </tr>
    <tr>
      <th>512696</th>
      <td>512696</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BA-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:44</td>
    </tr>
    <tr>
      <th>512697</th>
      <td>512697</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BA-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:46</td>
    </tr>
    <tr>
      <th>512698</th>
      <td>512698</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BB-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:48</td>
    </tr>
    <tr>
      <th>512699</th>
      <td>512699</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BB-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:50</td>
    </tr>
    <tr>
      <th>512700</th>
      <td>512700</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BC-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:52</td>
    </tr>
    <tr>
      <th>512701</th>
      <td>512701</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BC-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:54</td>
    </tr>
    <tr>
      <th>512702</th>
      <td>512702</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344D6-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:56</td>
    </tr>
    <tr>
      <th>512703</th>
      <td>512703</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.2</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344D6-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:58</td>
    </tr>
  </tbody>
</table>
<p>85296 rows × 16 columns</p>
</div>




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

    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


    co스케일변경


    /usr/local/lib/python3.7/site-packages/pandas/core/indexing.py:543: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>temp</th>
      <th>humidity</th>
      <th>pm_1</th>
      <th>pm25</th>
      <th>pm10</th>
      <th>co</th>
      <th>co2</th>
      <th>speed</th>
      <th>caron</th>
      <th>idx</th>
      <th>car_no</th>
      <th>detect_id</th>
      <th>type</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>427408</th>
      <td>427408</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>17</td>
      <td>17</td>
      <td>21</td>
      <td>0.178082</td>
      <td>640</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAC93-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:00</td>
    </tr>
    <tr>
      <th>427409</th>
      <td>427409</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>17</td>
      <td>17</td>
      <td>20</td>
      <td>0.178082</td>
      <td>640</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAC93-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:02</td>
    </tr>
    <tr>
      <th>427410</th>
      <td>427410</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>0.178082</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB8-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:04</td>
    </tr>
    <tr>
      <th>427411</th>
      <td>427411</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>0.178082</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB8-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:06</td>
    </tr>
    <tr>
      <th>427412</th>
      <td>427412</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>19</td>
      <td>0.178082</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB9-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:08</td>
    </tr>
    <tr>
      <th>427413</th>
      <td>427413</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>0.178082</td>
      <td>630</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACB9-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:10</td>
    </tr>
    <tr>
      <th>427414</th>
      <td>427414</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>15</td>
      <td>15</td>
      <td>18</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACBA-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:12</td>
    </tr>
    <tr>
      <th>427415</th>
      <td>427415</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACBA-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:14</td>
    </tr>
    <tr>
      <th>427416</th>
      <td>427416</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDD-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:16</td>
    </tr>
    <tr>
      <th>427417</th>
      <td>427417</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.4</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDD-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:18</td>
    </tr>
    <tr>
      <th>427418</th>
      <td>427418</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>13</td>
      <td>13</td>
      <td>16</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDE-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:20</td>
    </tr>
    <tr>
      <th>427419</th>
      <td>427419</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDE-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:22</td>
    </tr>
    <tr>
      <th>427420</th>
      <td>427420</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>13</td>
      <td>13</td>
      <td>15</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDF-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:24</td>
    </tr>
    <tr>
      <th>427421</th>
      <td>427421</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.3</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACDF-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:26</td>
    </tr>
    <tr>
      <th>427422</th>
      <td>427422</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFD-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:28</td>
    </tr>
    <tr>
      <th>427423</th>
      <td>427423</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFD-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:30</td>
    </tr>
    <tr>
      <th>427424</th>
      <td>427424</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFE-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:32</td>
    </tr>
    <tr>
      <th>427425</th>
      <td>427425</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFE-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:34</td>
    </tr>
    <tr>
      <th>427426</th>
      <td>427426</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>15</td>
      <td>15</td>
      <td>18</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFF-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:36</td>
    </tr>
    <tr>
      <th>427427</th>
      <td>427427</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AACFF-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:38</td>
    </tr>
    <tr>
      <th>427428</th>
      <td>427428</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD22-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:40</td>
    </tr>
    <tr>
      <th>427429</th>
      <td>427429</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>19</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD22-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:42</td>
    </tr>
    <tr>
      <th>427430</th>
      <td>427430</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD23-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:44</td>
    </tr>
    <tr>
      <th>427431</th>
      <td>427431</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.2</td>
      <td>16</td>
      <td>16</td>
      <td>18</td>
      <td>0.178082</td>
      <td>620</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD23-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:46</td>
    </tr>
    <tr>
      <th>427432</th>
      <td>427432</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>15</td>
      <td>15</td>
      <td>17</td>
      <td>0.178082</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3E-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:48</td>
    </tr>
    <tr>
      <th>427433</th>
      <td>427433</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3E-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:50</td>
    </tr>
    <tr>
      <th>427434</th>
      <td>427434</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3F-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:52</td>
    </tr>
    <tr>
      <th>427435</th>
      <td>427435</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD3F-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:54</td>
    </tr>
    <tr>
      <th>427436</th>
      <td>427436</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>16</td>
      <td>0.178082</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD40-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:56</td>
    </tr>
    <tr>
      <th>427437</th>
      <td>427437</td>
      <td>b33</td>
      <td>28.2</td>
      <td>60.1</td>
      <td>14</td>
      <td>14</td>
      <td>17</td>
      <td>0.178082</td>
      <td>610</td>
      <td>0.0</td>
      <td>0</td>
      <td>b42AAD40-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-29 00:00:58</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>512674</th>
      <td>512674</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D3F-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:58:59</td>
    </tr>
    <tr>
      <th>512675</th>
      <td>512675</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D3F-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:01</td>
    </tr>
    <tr>
      <th>512676</th>
      <td>512676</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4E-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:03</td>
    </tr>
    <tr>
      <th>512677</th>
      <td>512677</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4E-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:05</td>
    </tr>
    <tr>
      <th>512678</th>
      <td>512678</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4F-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:07</td>
    </tr>
    <tr>
      <th>512679</th>
      <td>512679</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D4F-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:09</td>
    </tr>
    <tr>
      <th>512680</th>
      <td>512680</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D50-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:11</td>
    </tr>
    <tr>
      <th>512681</th>
      <td>512681</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D50-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:13</td>
    </tr>
    <tr>
      <th>512682</th>
      <td>512682</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6A-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:15</td>
    </tr>
    <tr>
      <th>512683</th>
      <td>512683</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6A-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:17</td>
    </tr>
    <tr>
      <th>512684</th>
      <td>512684</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6B-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:19</td>
    </tr>
    <tr>
      <th>512685</th>
      <td>512685</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6B-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:21</td>
    </tr>
    <tr>
      <th>512686</th>
      <td>512686</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6C-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:23</td>
    </tr>
    <tr>
      <th>512687</th>
      <td>512687</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D6C-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:25</td>
    </tr>
    <tr>
      <th>512688</th>
      <td>512688</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D7B-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:27</td>
    </tr>
    <tr>
      <th>512689</th>
      <td>512689</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>925</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4633D7B-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:29</td>
    </tr>
    <tr>
      <th>512690</th>
      <td>512690</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634498-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:31</td>
    </tr>
    <tr>
      <th>512691</th>
      <td>512691</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634498-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:33</td>
    </tr>
    <tr>
      <th>512692</th>
      <td>512692</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634499-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:35</td>
    </tr>
    <tr>
      <th>512693</th>
      <td>512693</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b4634499-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:37</td>
    </tr>
    <tr>
      <th>512694</th>
      <td>512694</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b463449A-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:40</td>
    </tr>
    <tr>
      <th>512695</th>
      <td>512695</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b463449A-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:42</td>
    </tr>
    <tr>
      <th>512696</th>
      <td>512696</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BA-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:44</td>
    </tr>
    <tr>
      <th>512697</th>
      <td>512697</td>
      <td>b33</td>
      <td>27.0</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BA-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:46</td>
    </tr>
    <tr>
      <th>512698</th>
      <td>512698</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BB-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:48</td>
    </tr>
    <tr>
      <th>512699</th>
      <td>512699</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BB-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:50</td>
    </tr>
    <tr>
      <th>512700</th>
      <td>512700</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BC-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:52</td>
    </tr>
    <tr>
      <th>512701</th>
      <td>512701</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344BC-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:54</td>
    </tr>
    <tr>
      <th>512702</th>
      <td>512702</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344D6-1</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:56</td>
    </tr>
    <tr>
      <th>512703</th>
      <td>512703</td>
      <td>b33</td>
      <td>27.1</td>
      <td>41.3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027397</td>
      <td>930</td>
      <td>0.0</td>
      <td>0</td>
      <td>b46344D6-2</td>
      <td>72호4293</td>
      <td>G794210043F6</td>
      <td>Carnival</td>
      <td>2019-08-30 23:59:58</td>
    </tr>
  </tbody>
</table>
<p>85296 rows × 16 columns</p>
</div>




```python
test_set = tdf[feature].values.tolist()
```

# test co data / b33


```python
# 차트 
plt.figure(figsize=(25,5))
plt.plot(test_set, color='b')
```




    [<matplotlib.lines.Line2D at 0x7f6b086fbf28>]




![png](output_10_1.png)


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


![png](output_19_0.png)



![png](output_19_1.png)



```python
yPreds = model.predict(X_test)
yPred = np.argmax(yPreds, axis=1)
yTrue = np.argmax(Y, axis=1)

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

confusion_mat = metrics.confusion_matrix(yTrue, yPred)
print(confusion_mat)
print(confusion_mat.ravel())

print(metrics.classification_report(yTrue, yPred))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-78-683865bb6e21> in <module>
    ----> 1 yPreds = model.predict(X_test)
          2 yPred = np.argmax(yPreds, axis=1)
          3 yTrue = np.argmax(Y, axis=1)
          4 
          5 accuracy = metrics.accuracy_score(yTrue, yPred) * 100


    NameError: name 'model' is not defined

