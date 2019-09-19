---
title: "Preprocessing"
date: 2019-9-19 15:09:28 -0400
categories: DeepLearning
tags:
- Preprocessing
- LSTM
- RNN
- Predict model
---

# 데이터 쪼개기
* 전체 데이터셋에서 테스트 데이터만 분리, CSV로 저장
* predict할때, 빠르게 데이터 로드하기 위함
* dir 및 data list 예시 

```python
data_div = ['b37', 'b47', 'b43', 'b45', 'b3A', 'b2F', 'b42', 'b40', 'b48', 'b39',
       'b2E', 'b33', 'b36', 'b49', 'b4B', 'b31', 'b38', 'b35', 'b4A', 'b34',
       'b30', 'b3B', 'b41', 'b46', 'b3D', 'b44', 'b3E', 'b3C', 'b3F', 'b32']
```

```python
import os

# dataset2폴더가 있으면 그곳에 저장
# 없으면 detaset2폴더 만들고 그곳에 for문 이하 생성 데이터 저장

def saveAsCSV(idset):
    save_dir = '/data/dataset2/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    for i in idset:
        ff = traindata[traindata.id == i] # 조건이 맞는것만. 아래 이름으로 저장
        ff.to_csv(save_dir+"car_"+i+".csv", header=True, index=False)
        
```
 * saveAsCSV(data_div) #위 함수 살행, id리스트는 data_div로
