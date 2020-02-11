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

## 인덱스 초기화
```python
#인덱스 리셋 이거안하면 뒤에 concat할때 꼬인다.
df_C = df_C.reset_index(drop=True)
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

## 현재 경로 얻기
```python
  datapath = os.getcwd() 
  datapath
  ---
  Out '/home/jinyoungpark/KTR/preprocessing'
```
## TEST 데이터프레임 생성
```python

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
