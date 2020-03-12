---
title: "preprocess_LSTM"
date: 2020-02-17 15:09:28 -0400
categories: DeepLearning
tags:
- Development Environment
- tips
- preprocess
- LSTM
- RNN
- Generative Model
---

# LSTM 
## 보정모델
 - 목적 : 입력 데이터를 기준으로 고품질의 보정된 예측값을 얻는다.
 - 예시 : 저가 센서의 측정 데이터로 학습된 모델은 같은 저가 센서의 데이터를 input으로 받아 고품질의 측정기의 측정 수준에 data를 output으로 예측하게 한다.
 - 입력데이터 
  - 입력 단위 : 1분
  - feature : 4개
  
## 1. preprocess process

![](https://github.com/jypost/jypost.github.io/blob/master/img/LSTM_datamanager_jyp.png?raw=true)<br>

1. data load
 - 시계열 데이터 
  - 데이터 길이 확인,
  - 중간에 빈 row가 있는 경우들이 있음.
    - 학습에 쓸 데이터 구간인 경우,
     - 반드시 resampling('현재인터벌')을 해주어 빈 row를 nan값으로 채우고,
     - interpolate로 nan값을 채워야함
2. EDA data
  - visualizing
  - pattern
  - loss
  - balance
3. reshape
   - dType
   - dimension

## 2. preprocessing

```python
price1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
price2 = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]
sequence = 5 #인풋 X의 길이
predCount = 7 #출력할 Y값의 길이 ( 일주일치 예측하려면 7 넣으면 된다)

#인풋 데이터로 아웃풋을 만들때에는 input 과 output을 같은 데이터를 넣으면 됨
def InputSetTarget(Input, Output):
#     Input = X만들 데이터
#     Output = Y만들 데이터
    X = []
    Y = []
    Y_start_limit = len(Input)-predCount+1 #예측할 수 있는 마지막 값의 위치
    print('#예측 값 Y의 시작 위치', sequence1+1, '번째 부터')
    print('#예측 값 Y의 count ', predCount, ' 만큼씩')
    for i in range(len(Input)-sequence-predCount+1):
        Y_start = sequence + i #예측 값의 시작 위치
        X.append([Input[i+j] for j in range(sequence) if Y_start < Y_start_limit])
        Y.append([Output[Y_start + k] for k in range(predCount) if Y_start < Y_start_limit])

    print('#예측값 Y의 마지막 값의 위치', Y_start_limit, '번째 까지. 해당 번째의 시작 값은', Output[len(Output)-predCount])
    print('len(X): ', len(X), ' /  len(Y): ', len(Y))
    
    return X, Y

InputSetTarget(price1, price1)


# for i in range(len(price1)-sequence1):
#     X.append([price1[i+j] for j in range(sequence1)])
#     Y_start = sequence1 + i #예측 값의 시작 위치
#     Y_start_limit = len(price1)-predCount+1 #예측할 수 있는 마지막 값의 위치
#     if Y_start >= Y_start_limit: #예측 값의 시작 위치가 예측할 수 있는 마지막 값과 같거나 
#         Y_start = Y_start_limit-1
#     Y.append([price1[Y_start + k] for k in range(predCount)])


```



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
