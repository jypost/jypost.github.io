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
## 1. preprocess process
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
