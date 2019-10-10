---
title: "데이터 타입에 따른 차트 비교 "
date: 2019-10-10 15:09:28 -0400
categories: DeepLearning
tags:
- Pandas
- EDA
- Preprocessing
- Data Visualizing
- 데이터 시각화
- Graph
- chart
- Tips
---

### 데이터 타입과 시계열 차트 
  - 같은 에

같은 
같은 데이터를 Visualizing 할때,
데이터 타입이 DataFrame이냐 List냐 에따라 아래와 같이 차트가 다르게 나타난다
이유는 아직 모르겠다 더 파보면 알겠지


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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
plt.style.use('bmh')

import keras 
keras.__version__
```

    Using TensorFlow backend.

    '2.2.4'



# test data load & Preprocessing

### 1. test 데이터 불러오기 df.csv


```python
#검증데이터
testdata_path = '/data_0905/df.csv'
testdata = pd.read_csv(testdata_path)
print('\n testdata shape :', testdata.shape)
```

    
     testdata shape : (2499088, 16)



```python
testdata.id.value_counts()
```




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
    b3A    85368
    b4A    85368
    b46    85306
    b33    85296
    b42    85242
    b37    85048
    b44    84794
    b3C    66772
    b47    38864
    Name: id, dtype: int64




```python
choice_id_list = ['b33','b4A','b32','b36','b41','b42']

df = testdata
df = df[df['id'].isin(choice_id_list)]
value_ids = df['id'].value_counts()
print('df.shape : ',df.shape)
print('value_ids.index :',value_ids.index)
```

    df.shape :  (512998, 16)
    value_ids.index : Index(['b41', 'b36', 'b32', 'b4A', 'b33', 'b42'], dtype='object')



```python
df = df[['id','pm25','co']] #해당 컬럼명만 선택
```


```python
#원본 데이터 차트보기 (스케일 먹이기 전)
df_pm25 = df['pm25'].values.tolist()

plt.figure(figsize=(30,4))
ax = plt.gca()
ax.set_facecolor((1, 1, 1))
plt.plot(df_pm25, marker='o', markersize=0.7, linestyle='-', color='b', alpha=0.4)
plt.title('original data : list Type')

print('원본 데이터 셋 /List타입 길이 확인 : ',len(df_pm25))
```

    원본 데이터 셋 /List타입 길이 확인 :  512998



![png](https://github.com/jypost/jypost.github.io/blob/master/img/output_6_1.png?raw=true)



```python
#원본 데이터 차트보기 (스케일 먹이기 전)
# train_set2 = df[feature].values.tolist()
# DataFrame으로 차트 보기
df_pm25 = df['pm25']
df_pm25.shape

plt.figure(figsize=(30,4))
ax = plt.gca()
ax.set_facecolor((1, 1, 1))
plt.plot(df_pm25, marker='o', markersize=0.7, linestyle='-', color='b', alpha=0.4)
plt.title('original data : DataFrame Type')

print('원본 데이터 셋 / DataFrame타입 shape 확인 : ',df_pm25.shape)
```

    원본 데이터 셋 / DataFrame타입 shape 확인 :  (512998,)



![png](https://github.com/jypost/jypost.github.io/blob/master/img/output_7_1.png?raw=true)

