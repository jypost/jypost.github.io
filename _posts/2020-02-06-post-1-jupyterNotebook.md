---
title: "개발 환경 설정들"
date: 2020-02-06 15:09:28 -0400
categories: DeepLearning
tags:
- Development Environment
- jupyter notebook
- python 
- pip
---

# jupyter 
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
