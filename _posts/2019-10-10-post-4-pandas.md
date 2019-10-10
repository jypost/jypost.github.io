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
## Dataframe 에서 특정 컬럼의 값에 따른 데이터 분류

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