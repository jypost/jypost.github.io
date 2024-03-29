---
title: "Pandas 정리"
date: 2019-10-10 15:09:28 -0400
categories: DeepLearning
tags:
- Preprocessing
- Pandas
- 데이터 
- Tips
---

# Pandas 주요 기능 정리
## 1. Df에서 특정 컬럼의 반복되는 값에 따른 데이터 분류

```python
#임의의 데이터 프레임 생성
df1 = pd.DataFrame(data=np.array([['b22', 17, 16, 16, 16], 
                                  ['b21', 5, 6, 1, 1], 
                                  ['b20', 8, 9, 1, 1]
                                 ]), columns=['A', 'B', 'C','d','e'])

#드롭 또는 셀렉트할 str list (일단 한개만 테스트)
droplist = ['b22']

# df1 = df1[~df1['A'].isin(droplist)] #A에서 droplist가 포함되는 것 제거
df1 = df1[df1['A'].isin(droplist)] #A에서 droplist가 포함되지 않은 모든것 제거

df1
```

<style type="text/css">
.tg {border-collapse:collapse;
  border-spacing:0;
  border-color:#ccc;
  height:1.4rem;
  }
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-baqh{text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-baqh"></th>
    <th class="tg-baqh">a</th>
    <th class="tg-baqh">b</th>
    <th class="tg-baqh">c</th>
    <th class="tg-baqh">d</th>
    <th class="tg-baqh">e</th>
  </tr>
  <tr>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">b22</td>
    <td class="tg-baqh">17</td>
    <td class="tg-baqh">16</td>
    <td class="tg-baqh">16</td>
    <td class="tg-baqh">16</td>
  </tr>
</table>



## 2. 특정 이름을 가진 column을 제거
```python
'''
    특정 이름을 가진 column을 제거할때 응용이 많은 코드
'''

data = {
            'pm25' : np.arange(5),
            'co' : np.arange(5, 10),
            'pm10' : np.arange(10, 15),
}
df = pd.DataFrame(data)
df
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
      <th>pm25</th>
      <th>co</th>
      <th>pm10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>7</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>9</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>


```python
'''
    column명의 앞 두(N) 글자를 조건으로 
    선택 또는 선택해제된 Data 파일 생성
'''

data = {
            'pm25' : np.arange(5),
            'co' : np.arange(5, 10),
            'pm10' : np.arange(10, 15),
}

df = pd.DataFrame(data=data)
cols = [c for c in df.columns if c.lower()[:2] != 'pm']
df = df[cols]
df
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
      <th>co</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




## 특정 컬럼에서 같은 값이 중복되는 갯수 확인
```python
df1 = pd.DataFrame(data=np.array([['b22', 17, 16, 16, 16],
                                  ['b22', 17, 16, 16, 16],
                                  ['b22', 17, 16, 16, 16],
                                  ['b22', 17, 16, 16, 16],
                                  ['b22', 17, 16, 16, 16],
                                  ['b21', 5, 6, 1, 1], 
                                  ['b21', 5, 6, 1, 1], 
                                  ['b21', 5, 6, 1, 1], 
                                  ['b21', 5, 6, 1, 1],                                   
                                  ['b20', 8, 9, 1, 1],
                                  ['b20', 8, 9, 1, 1],
                                  ['b20', 8, 9, 1, 1],
                                  ['b20', 8, 9, 1, 1]                                  
                                 ]), columns=['A', 'B', 'C','d','e'])
# 특정 컬럼에서 같은 값이 중복되는 갯수 확인
df1.A.value_counts()
```

  b22    5<br>
  b20    4<br>
  b21    4<br>
  Name: A, dtype: int64
 


## 특정 컬럼에서 반복되는 값(id) 축출, 리스트로

```python

rowdf.sen_id.value_counts()
rowdf.sen_id.value_counts().index

out ======
Index(['ktr0052', 'ktr0032', 'ktr0033', 'ktr0039', 'ktr0042', 'ktr0043',
       'ktr0044', 'ktr0040', 'ktr0035', 'ktr0055', 'ktr0041', 'ktr0038',
       'ktr0054', 'ktr0037', 'ktr0050', 'ktr0051', 'ktr0047', 'ktr0053',
       'ktr0045', 'ktr0031', 'ktr0046', 'ktr0056', 'ktr0048', 'ktr0036',
       'ktr0057', 'ktr0059', 'ktr0062', 'ktr0061'],
      dtype='object')

sensor_list = rowdf.sen_id.value_counts().index.tolist()
sensor_list.sort()
sensor_list[:5]

out ======
['ktr0031', 'ktr0032', 'ktr0033', 'ktr0035', 'ktr0036']


```










## 데이터 프레임 발라내기 column select 
```python

#사용할 column만
df_b4A = df_b4A[['time','id','pm25','co','type']]
df_b36 = df_b36[['time','id','pm25','co','type']]
print(df_b4A.shape)
print(df_b36.shape)


data = {
            'pm25' : np.arange(5),
            'co' : np.arange(5, 10),
            'pm10' : np.arange(10, 15),
}


df = pd.DataFrame(data=data)

# 데이터 프레임 이렇게 잡으면 array로 잡힌다.
ddd = df['co']
ddd

# 데이터 프레임 이렇게 잡으면 데이터프레임으로 잡힌다.
ddd1 = df[['co']]
ddd1

# column 내에서 최대값


```
## 데이터 프레임 복사할때
- 데이터 프레임 복사할떄 copy()로 하면 copy본을 조작해도 원본에 영향없다.

```python

영향 없음
df = df_X3.copy()

영향 줌
df = df_X3[:]
df = df_X3

import copy
test_list = df_list.copy() #얕은 복사
# # test_list = copy.deepcopy(df_list) #깊은 복사

```

## 특정프레임에 함수 적용, Apply로
- 시간 표기법을 바꾸는 함수 적용.

```python

bitcoin_df.head()
out: 
	close	date	high	low	open	quoteVolume	volume	weightedAverage
1847	7944.855000	1583884800	7955.554909	7868.006470	7891.460535	113.291921	8.971284e+05	7918.732692
1846	7891.259671	1583798400	8145.000000	7736.767169	7930.300000	2701.215817	2.146016e+07	7944.628704
1845	7930.300000	1583712000	8175.426929	7633.000000	8032.417896	5515.072726	4.349578e+07	7886.709768
1844	8032.017896	1583625600	8892.767431	7999.000000	8892.767431	5683.044831	4.811151e+07	8465.797212
1843	8891.517537	1583539200	9182.605000	8840.936923	9132.390961	1825.009552	1.651056e+07	9046.833132

from time import localtime, strftime
def changeTime(nowdate):
#     now = nowdate
    local_tuple = localtime(nowdate)
    time_format = '%Y-%m-%d %H:%M:%S'
    return strftime(time_format, local_tuple)

bitcoin_df['date'] = bitcoin_df['date'].apply(changeTime)
bitcoin_df.head()
out:

	close	date	high	low	open	quoteVolume	volume	weightedAverage
1847	7944.855000	2020-03-11 09:00:00	7955.554909	7868.006470	7891.460535	113.291921	8.971284e+05	7918.732692
1846	7891.259671	2020-03-10 09:00:00	8145.000000	7736.767169	7930.300000	2701.215817	2.146016e+07	7944.628704
1845	7930.300000	2020-03-09 09:00:00	8175.426929	7633.000000	8032.417896	5515.072726	4.349578e+07	7886.709768
1844	8032.017896	2020-03-08 09:00:00	8892.767431	7999.000000	8892.767431	5683.044831	4.811151e+07	8465.797212
1843	8891.517537	2020-03-07 09:00:00	9182.605000	8840.936923	9132.390961	1825.009552	1.651056e+07	9046.833132

```



## array를 list로 
```python

#array를 list로
ddd = ddd.to_list()
ddd

```

## ndarray를 list로 
```python

#df를 list로
ddff = list(bbb['tm'].values)
ddff

```

## ndarray를 2차원 > 1차원으로  
```python


In [1]: import numpy as np
...: x = np.arange(12).reshape(3, 4)
...: x
...:

Out[1]:
array([[ 0, 1, 2, 3],
        [ 4, 5, 6, 7],
        [ 8, 9, 10, 11]])

np.ravel(x, order='C') # by default
Out[2]: array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

np.ravel(x, order='F')
Out[3]: array([ 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

order=K 도있는데, 메모리 발생순
```


## threshold 구하기

```python
list_err = err['pm25_dif'] # 정상차량 스퀘어 에러 리스트 
mean, std = np.average(list_err), np.std(list_err) #여기서 mean, std 축출
threshold = mean + (3*std) #아래 수식으로 thresfhold 연산
print('mean', mean)
print('std', std)
print('threshold', threshold)

out:
mean 0.34924421070164924
std 49.3638057713741
threshold 148.44066152482395

```

## 소수점 반올림

```python
#round(수, 자릿수)
print(round(end_time - start_time, 4))

#정수로
x=35.6
int(x)

```

## 소수점 표시

```python
number = 4
num = '%.'+str(number)+'f' # 표시할 자리수
ss = num % 34.293875
ss

out:
'34.2939'

```



## 시간측정 1

```python
import timeit
start = timeit.default_timer()

# 실행 코드

stop = timeit.default_timer()
print(stop - start)
```

## 시간측정 2 decorator 사용

```python

import time

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print('실행시간 : ',round(end_time - start_time, 4),'초')
        return result
    return wrapper_fn

#아래처럼 사용, 함수위에 써주면 된다. 함수 실행시 위의 함수가 함꼐 호출된다.
@logging_time
def my_func1():
    print("시간측정")

#데코레이터에 사용되는 함수는 반드시 적용되어질 대상 함수보다 먼저 정의되어야 한다.
2.적용은 함수 외에도 클래스도 가능

```

## list의 원소인 str을 변수로 데이터프레임 반복 생성 (엄청 많이씀)
```python

#리스트를 변수로 생성 
selectlist = ['A','B','C']

# 아래 함수는 ID가 여러개인 전체 데이터 프레임에서 
# 각 아이디별로 데이터 프레임을 생성하는 함수 입니다.
# 생성할 ID_list와 ID_list에서 생성된 각각의 변수에 저장할 data를 인자로 받습니다.

def listTodf(list, data): #list를 입력으로 해당 리스트의 변수에 데이터프레임 생성 / data는 생성할 데이터 프레임 전체
    mod = sys.modules[__name__]
    num = 0 
    for i in list: #i는 변수명
        x = '{}'.format(i) #리스트의 str로 변수 생성 (각각)
        v = data[data['id'].isin([list[num]])]
        setattr(mod, x, v) #setattr(object, name, value) 조건으로 데이터프레임 생성
        print(x+' DataFrame이 생성되었습니다. '+x+'.shape :', v.shape)
        num += 1


# 특정 ID만 찍어서 데이터 프레임으로
fff = df[df['id'].isin(['b41'])]
fff.shape
(85800, 16)
```

## 특정범위 선택
```python

threshold=44.07

df_A_count = df_A_final[(df_A_final['pm'] > threshold)]
df_C_count = df_C_final[(df_C_final['pm'] > threshold)]
df_M_count = df_M_final[(df_M_final['pm'] > threshold)]


# Bot50
BOTTOM = df_all[(df_all['pm'] > 0.) & (df_all['pm_dif'] >= 0.01)]
Bot50 = BOTTOM.sort_values(by='pm_dif', ascending=True).head(50)

# Top50
Top50 = df_all.sort_values(by='pm_dif', ascending=False).head(50)

```

## 특정날짜범위 선택
```python
sen_tm    172746 non-null datetime64[ns]

testDfWeb = webdf[(webdf['sen_tm'] < '2019-11-14 00:01') & (webdf['sen_tm'] > '2019-10-14 00:00')]
```

## 특정날짜 구간 나누기
```python

df_Y1 = df_Y[dt.datetime(2019, 10, 12):dt.datetime(2019, 11, 15, 23, 59)]
#년,월,일,시,분...
```


## 날짜기준 정렬
```python
testDfWeb.sort_values(by=['sen_tm'], inplace=True, ascending=True)

out:

sen_id	sen_tm	temp	humid	PM10	PM25
110616	SGO-003	2019-11-14 23:59:00	29.6	48.1	27.6	12.8
110617	SGO-003	2019-11-14 23:58:00	29.6	48.1	27.6	12.8
110618	SGO-003	2019-11-14 23:57:00	29.6	48.1	24.6	13.0
110619	SGO-003	2019-11-14 23:56:00	29.6	48.1	24.6	13.0
110620	SGO-003	2019-11-14 23:55:00	29.6	48.1	24.0	13.2

```

## Series Type 정렬

```python
test = series.sort_index()

```

## Time Series chart index

```python
#time Series는 index data type이 datetime이어야 차트에 날짜 찍힌다.
df.index.dtype # index type 확인 ('o') object type 임
df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
#df = df.set_index() index순으로 데이터 정렬
print('index.dtype : ', df.index.dtype)

out:
index.dtype :  datetime64[ns]

```

## 결측값에 str데이터('-')가 입력되어 있을때, NaN값으로 대체하기

```python
df['PM10'] = df['PM10'].replace('-', np.nan)

out:
   sen_id	          sen_tm	temp	humid	PM10	PM25  
0	SGO-016	2020-02-12 10:38	  NaN	NaN	NaN	NaN
1	SGO-016	2020-02-12 10:37	  NaN	NaN	NaN	NaN
2	SGO-016	2020-02-12 10:36	  NaN	NaN	NaN	NaN
3	SGO-016	2020-02-12 10:35	  NaN	NaN	NaN	NaN
4	SGO-016	2020-02-12 10:34	  NaN	NaN	NaN	NaN
```

## 데이터 타입 변경

```python
#데이터 타입 변경 temp이후 전체row를 모두 float64로
df.iloc[:,2:] = df.iloc[:,2:].astype('float64')

out:
   sen_id	          sen_tm	temp	humid	PM10	PM25  
0	SGO-016	2020-02-12 10:38	  NaN	NaN	NaN	NaN
1	SGO-016	2020-02-12 10:37	  NaN	NaN	NaN	NaN
2	SGO-016	2020-02-12 10:36	  NaN	NaN	NaN	NaN
3	SGO-016	2020-02-12 10:35	  NaN	NaN	NaN	NaN
4	SGO-016	2020-02-12 10:34	  NaN	NaN	NaN	NaN

```

## 데이터프레임 정렬

```python
def df_column_set(data): #data는 최종 데이터 프레임
    df_reloc = pd.DataFrame(data=data, columns=['time',
                                                'id',
                                                'pm25',
                                                'co',
                                                'pm25_pred',
                                                'co_pred',
                                                'pm25_dif',
                                                'co_dif',
                                                'type',
                                               ])
    return df_reloc
```
## 데이터 프레임 NAN값 갯수 확인
```python
df.isnull().sum()
---
output
col1    0
col2    0
col3    0
col4    0
col5    0
dtype: int64
```

## 특정 값 다른값으로 변경 '-' -> 0
```python

ddd = [c for c in testDfWeb['PM25'] if c.lower()[:2] == '-']
ddd
out : ['-', '-', '-', '-', '-', '-', '-', '-']

#NAN값이 '-'로 입력되어있어 float64 변환이 안되어 아래와 같이 변환함

testDfWeb['PM25'] = testDfWeb['PM25'].replace('-', 0)
testDfWeb['PM25'] = testDfWeb['PM25'].astype('float64')

```

## 데이터프레임 인덱스 값 변경: 시간단위>분단위 일 경우, 변경에 따른 없는 값은 Nan으로
```python

#nan 이포함된 테스트 데이터프레임 생성 
test_datas = {'pm25' : np.arange(0,5),
             'tm' : pd.date_range('2019-01-01',periods=5, freq='3T')
             }
test_df = pd.DataFrame(data=test_datas)
test_df = test_df.set_index('tm').sort_index()

#특정값 nan으로 변경

test_df['pm25'] = test_df['pm25'].replace([c for c in test_df['pm25'] if c % 2 == 0], np.nan)

# for i in test_df['pm25']:
#     if i % 2 == 0:
#         test_df['pm25'] = test_df['pm25'].replace(i, np.nan) 

print(len(list(test_df.index)))

test_df

```
out put 
```python
	pm25
tm	
2019-01-01 00:00:00	NaN
2019-01-01 00:03:00	1.0
2019-01-01 00:06:00	NaN
2019-01-01 00:09:00	3.0
2019-01-01 00:12:00	NaN
```

데이터 프레임 인덱스 간격 변경 3분단위 > 1분 단위 (없는 값은 nan으로 입력됨 )
```python
range_date = pd.date_range('2019-01-01', '2019-01-01 00:10', freq='1T')
out_df = pd.DataFrame(index = range_date, data = test_df)
out_df
```
out put
```python
	pm25
2019-01-01 00:00:00	NaN
2019-01-01 00:01:00	NaN
2019-01-01 00:02:00	NaN
2019-01-01 00:03:00	1.0
2019-01-01 00:04:00	NaN
2019-01-01 00:05:00	NaN
2019-01-01 00:06:00	NaN
2019-01-01 00:07:00	NaN
2019-01-01 00:08:00	NaN
2019-01-01 00:09:00	3.0
2019-01-01 00:10:00	NaN
```

## resampling, 시계열 data 비거나 중복되는 row 맞춤
```python
df = df.resample('1T').mean()
```
<span style="color:#363636; font-size:2pt; line-height: 0.5pt; margin-left: 1.5rem;">1분 간격 데이터로 Resampling<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * freq 인수값 : <br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * s: 초<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * T: 분<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * H: 시간<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * D: 일(day)<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * B: 주말이 아닌 평일<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * W: 주(일요일)<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * W-MON: 주(월요일)<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * M: 각 달(month)의 마지막 날<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * MS: 각 달의 첫날<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * BM: 주말이 아닌 평일 중에서 각 달의 마지막 날<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * BMS: 주말이 아닌 평일 중에서 각 달의 첫날<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * WOM-2THU: 각 달의 두번째 목요일<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * Q-JAN: 각 분기의 첫달의 마지막 날<br></span>
<span style="color:#363636; font-size:2pt; margin-left: 1.5rem;">    * Q-DEC: 각 분기의 마지막 달의 마지막 날<br></span>


## interpolate(), NaN값 채우기. 보통, 위에 resampling하고 interpolate 한다.

```python
df_interpolate = df.interpolate()
```

## 인덱스 초기화
```python
#인덱스 리셋 이거안하면 뒤에 concat할때 꼬인다.
df_C = df_C.reset_index(drop=True)
```

## 인덱스 기준으로 데이터 정렬, 오름차순, 내림차순
```python
df = df.sort_index()
df = df.sort_index(ascending=False)

#아래는 index로 설정하면서 차순 정렬
dfs = dfs.set_index('date').sort_index(ascending=False)

```

## 인덱스 datetime으로 변경 
```python
#time index 설정
df = df.set_index('tm') #tm컬럼을 index로 변경

```

## 인덱스에 str형태때문에 datetime64로 변환 안되는 case

- time column의 dtype이 object이고, str에 24가 들어있는경우
- time columns의 datetime64로 변환이 안됨. 아래 코드 쓰면 됨

```python

def my_to_datetime(date_str):
    if date_str[11:13] != '24':
        return pd.to_datetime(date_str, format='%Y-%m-%d:%H')
 
    date_str = date_str[0:11] + '00'
    return pd.to_datetime(date_str, format='%Y-%m-%d:%H') + dt.timedelta(days=1)
    
def newAKdf2():
    ak = pd.concat(air_list, axis=0) #위에서 받은 로드 리스트 배열을 하나의 df로 합
    ak.tm = ak.tm.apply(my_to_datetime) #tm컬럼 24:00분 -> 00:00 분으로하고 1일 추가함 
    ak = ak.set_index('tm').sort_index() #tm column을 index로 변경하고, 정렬함
    print('에어코리아', ak.shape, 'PM25 결측', ak['PM25'].isnull().sum(),'개')
    #ak.index = ak.index.astype('datetime64[ns]') 이거 안됨. tm에 24가 있어서 converting 이 안됨.
    ak.info()
    return ak
    
df = newAKdf2() #  사용하면됨

```


## 인덱스 data Type 변경 
```python
# index가 datetime 일 경우, dtype이 datetime이어야 차트에서 X축에 시간표시됨
df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
print('index.dtype : ', df.index.dtype)

out :
index.dtype :  datetime64[ns]
```

## 인덱스 name 변경 
```python
df.index.name = 'tm'
```


## 차트, matplot legend font argument 

- 한글 폰트 적용할때, legend부분만 깨져서 fontproperties로 넘기려하면 에러난다<br>
- 아래와 같이 prop로 넘기면 된다

```python

#한글폰트
path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fontprop = fm.FontProperties(fname=path, size=16)

font_name = fm.FontProperties(fname=font_path).get_name()
plt.legend(['A','B'],prop=fontprop)

```

## 차트, title, x label 간격

```python
plt.title(feature, fontproperties=fontprop, pad= 17)
plt.xlabel(datatate, labelpad=20)

```

## 차트, 특정 구간 강조

```python
plt.plot(data['PM25'], color='#4287f5', marker='o', markersize=0.7, alpha=1)
plt.plot(data['PM10'], color='#eb8b3d', marker='o', markersize=0.7, linestyle='-', alpha=0.5)      
span_start = datetime(2020, 10, 13, 14)
span_end = datetime(2020, 10, 14, 13)
plt.axvspan(span_start, span_end, facecolor='r', alpha=0.3)

```


## 차트에 한글 폰트 적용

- 폰트 확인

```python
font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
font_list

output:
['/fonts/truetype/nanum/NanumSquareEB.ttf',

```
- 폰트 설치 ( 한글폰트 없을경우 )

```python

#한글 폰트가 없을 경우 아래 명령으로 나눔폰트 설치 가능 
! sudo apt-get install -y fonts-nanum fonts-nanum-coding fonts-nanum-extra

```
- 한글 적용

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
%matplotlib inline   

path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fontprop = fm.FontProperties(fname=path, size=16)

def showshow(Y, title, hlineN):
    plt.figure(figsize=(20, 4))
    plt.rcParams['axes.unicode_minus'] = False #음수 표시 가능하게 
    ax = plt.gca()
    ax.set_facecolor((1, 1, 1))
#     ax.grid(False)
    ax.set_ylabel(" prince $",x=0)
    ax.set_xlabel(" 비트코인",y=0, fontproperties=fontprop)
    ax.yaxis.set_label_coords(-0.05,0.5)
    plt.rcParams["axes.facecolor"] = 'white'
    plt.rc('font', family='NanumGothicOTF')
#     plt.rcParams["font.family"] = u'AppleGothic' #한글 깨질때 
#     plt.plot(Y_hat, marker='o', markersize=1, linestyle='None', color='c')
    plt.plot(Y, marker='o', markersize=0.1, linestyle='-', color='#198AEC', alpha=1)
    plt.xticks(np.arange(0, df.shape[0], step=365),["s_{:0<2d}".format(x) for x in df['date'].values], 
#                fontproperties=BMDOHYEON, 
               fontsize=10, 
               rotation=0)
    plt.axhline(y=hlineN, color='r', linewidth=1, alpha=0.7)
    plt.legend(['Price','current Price'])
    plt.legend(['에어코리아', 'KTR'], prop=fontprop)
    plt.title(str(title), fontproperties=fontprop)
    plt.show()
```



## 데이터 로컬 저장 

- csv로 저장 

```python
datapath = os.getcwd() 

def saveAsCSV(data, savename):
    save_dir = datapath+'/data/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    data.to_csv(save_dir+savename+".csv", header=True, index=False)
```

- excel로 저장 

```python
datapath = os.getcwd() 

def saveAsxls(data, name):
    save_dir = datapath+'/data/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    data.to_excel(save_dir+name+".xls", sheet_name='Sheet1', index=False)
```

## 데이터 로컬 불러오기 

```python
# data load 
name = 'data'
path = './' #현재폴더
df = pd.read_csv(path+name+'.csv', index_col='인덱스로 쓸 컬럼 명')


data_path = '/home/data'
df = pd.read_excel(data_path + '/data.xls')
```

## 현재 경로 얻기
```python
  datapath = os.getcwd() 
  datapath
  ---
  Out '/home/preprocessing'
```

## columns 선택
```python
#컬럼선택
df = traindata[['센서Code', 'PM10 (μg/㎥)', 'PM2.5 (μg/㎥)', 'Temp (℃)', 'Humid (%)', 'Time']]
```

## columns 변경
```python
#컬럼명 변경
df.columns = ['sensorid', 'PM10', 'PM25', 'temp', 'humid', 'tm']
```

## TEST 데이터프레임 X,Y mapping set 
```python
X = np.arange(0,10)
Y = np.arange(0,10)
print(X)
print(Y)

def InputSetTarget(data_X, data_Y, sq):
    X = []
    Y = []
    for i in range(len(data_X)-sq+1):
        X.append([data_X[i+j] for j in range(sq)])
        Y.append(data_Y[sq+i-1])
#     print('len(X) :',len(X))
#     print('len(Y) :',len(Y))
    return X, Y

x, y = InputSetTarget(X,Y,3)
print(x)
print(y)

out========
[0 1 2 3 4 5 6 7 8 9]
[0 1 2 3 4 5 6 7 8 9]
[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
[2, 3, 4, 5, 6, 7, 8, 9]
```

## TEST 데이터프레임 생성
```python
#시계열 데이터 샘플 생성
test_datas = {'pm25' : np.arange(0,50),
             'tm' : pd.date_range('2019-01-01',periods=50, freq='1T')
             }
test_ktr = pd.DataFrame(data=test_datas)
test_ktr = test_ktr.set_index('tm').sort_index()

test_ktr2 = test_ktr.resample('2T').first() #2분

print(test_ktr.head())
print(test_ktr2.head())


#시계열 데이터 샘플 랜덤 생성 
# generate time series index
range = pd.date_range('2019-12-19', '2019-12-20', freq='2min')
df = pd.DataFrame(index = range)[:20]

# add 'price' columm using random number
np.random.seed(seed=1004) # for reproducibility
df['price'] = np.random.randint(low=10, high=100, size=20)

# add 'amount' column unsing random number
df['amount'] = np.random.randint(low=1, high=5, size=20)
print('Shape of df DataFrame:', df.shape)
```
## 데이터 프레임 합치기
```python
#가로축으로 입력 순서대로 합치기
newdf = pd.concat([df1, df2], axis=1)
newdf
```
### 데이터 프레임 concat 할때 유의사항
  - axis=1 로 concat할때, 두 DataFrame간 index가 안맞으면, NAN값이 생기므로 주의할것
  
## 인덱스 순서 바꾸기
```python
```

## 데이터 프레임 column 삭제
-axis=0은 row을, axis=1은 column을 의미

```python
df = df.drop("A", axis=1)
# df = df.drop(columns="A")
print(df)

# 출력
#      B       C  D  E
# c1  0  300  0  0
# c2  0  301  0  0
# c3  0  302  0  0
# c4  0  303  0  0
# c5  0  304  0  0

# 여러개 삭제시
df = df.drop(["A", "C"], axis=1)
# df = df.drop(columns=["A", "C"])
print(df)

# 출력
#      B  D  E
# c1  0  0  0
# c2  0  0  0
# c3  0  0  0
# c4  0  0  0
# c5  0  0  0
```

## 데이터 프레임 row 삭제
-axis=0은 row을, axis=1은 column을 의미

```python
df = df.drop("c1") # df = df.drop("c1", axis=0)
# df = df.drop(index="c1")
print(df)

# 출력
#          A  B       C  D  E
# c2  101  0  301  0  0
# c3  102  0  302  0  0
# c4  103  0  303  0  0
# c5  104  0  304  0  0

```
-index가 int일때
```python
df = df.drop(0) # df = df.drop(0, axis=0)
# df = df.drop(index=0)
print(df)

# 출력
#          A  B       C  D  E
# 1  101  0  301  0  0
# 2  102  0  302  0  0
# 3  103  0  303  0  0
# 4  104  0  304  0  0

```

## 데이터 프레임을 리스트로 만들고, 해당 리스트를 for문을 돌려 편집하면, 각 원소의 값에 변화가 없다
## 꼭 확인
```python
아래처럼 하면 된다. 나중에 정리

#아이디별 생성한 데이터 프레임 원본 리스트
selectdf = [b36, b41, b42, b33, b4A, b32]

# 인덱스 리셋 리스트 (변수생성할떄 씀)
resetdf = ['b36','b41','b42','b33','b4A','b32']

# 1. 컬럼을 먼저 정리하고, 
def reset_column(resetdf, selectdf): #resetdf = 생성변수 str 리스트 / selectdf는 원본 데이터 프레임 리스트
    mod = sys.modules[__name__]
    num = 0
    for i in resetdf: #i는 생성될 변수명 
        x = '{}'.format(i) #리스트의 str로 변수 생성 (각각)
        v = selectdf[num][['time','id','pm25','co','type']] 
        setattr(mod, x, v) #setattr(object, name, value) 조건으로 df 생성
        print('\n',v.head())
        num += 1        

# 2. 인덱스를 리셋한다.         
def reset_index(resetdf, selectdf): #resetdf = 생성변수 str 리스트 / selectdf는 원본 데이터 프레임 리스트
    mod = sys.modules[__name__]
    num = 0
    for i in resetdf: #i는 생성될 변수명 
        x = '{}'.format(i) #리스트의 str로 변수 생성 (각각)
        v = selectdf[num].reset_index(drop=True)
        setattr(mod, x, v) #setattr(object, name, value) 조건으로 df 생성
        print('\n',v.head())
        num += 1

# 1. 컬럼을 먼저 정리하고, 
reset_column(resetdf, selectdf)
selectdf = [b36, b41, b42, b33, b4A, b32] # 아래 for문으로 데이터 조작위해서 

# 2. 인덱스를 리셋한다. 
reset_index(resetdf, selectdf)

b42.head()

```

# jupyter 환경설정
## 1. 설치된 패키지 확인 

```python
  pip freeze
```
```python
json5==0.8.5
jsonschema==3.0.2
jupyter-client==5.3.3
jupyter-core==4.5.0
jupyter-tensorboard==0.1.10
jupyterhub==1.0.0
jupyterlab==1.0.2
jupyterlab-server==1.0.0
jupytext==1.2.4
Keras==2.3.1
Keras-Applications==1.0.6
Keras-Preprocessing==1.0.5
```

## 2. 패키지 설치
 
 - plotly(설치패키지명 예시)
 
```python
  # !sudo pip install plotly
```

## 3. 다른 경로 파일 import
 
 - 동일 경로가 아닌 상위 경로의  .py 파일을 불러올때 아래와 같이 path를 추가해주면 된다
 
```python
import os 
import sys
#공통모듈 불러오기
sys.path.append('/home/data/common module/')
from datamanager import *
```


## 4. 케라스에서 만든 모델, 레이어/shape확인/모델구조 이미지저장
 
 
```python
  # LSTM모델 예시
  ### LSTM 모델
np.random.seed(0)
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(60, 1))) # return_sequences=True,
model.add(LSTM(32))
model.add(Dense(1, activation='relu')) 
model.compile(loss='mse', optimizer='Adam')
model.summary()

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

%matplotlib inline
plot_model(model,  show_shapes=True,to_file='model.png')
  
```

## 5. 터미널에서 쥬피터 파일 압축/압축풀기
 
 - plotly(설치패키지명 예시)
 
현재 폴더 보기
입력창: ls

압축 : zip -r backupALL.zip back_up_ALL_1023
Zip / -r / 압축될 파일 폴더명.zip / 압축될 파일 폴더

압축 풀때
unzip test.zip
unzip / 압축풀 파일명

