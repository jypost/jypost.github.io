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
