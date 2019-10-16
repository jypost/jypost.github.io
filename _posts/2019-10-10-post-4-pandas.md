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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b22</td>
      <td>17</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



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


```

## array를 list로 
```python

# array를 list로
ddd = ddd.to_list()
ddd

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



## 인덱스 초기화
## TEST 데이터프레임 생성
## 데이터 프레임 합치기
### 데이터 프레임 concat 할때 유의사항
  - axis=1 로 concat할때, 두 DataFrame간 index가 안맞으면, NAN값이 생기므로 주의할것
## 인덱스 순서 바꾸기
## 데이터 프레임 column 삭제
Drop
## 데이터 프레임 NAN값 갯수 확인

