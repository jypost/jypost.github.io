---
title: "Validation processing2"
date: 2019-9-24 15:09:28 -0400
categories: DeepLearning
tags:
- Preprocessing
- validation model
- test model
- validation process
---


# ☝🏻Process summary
#### <span style="color:black;">1.test data load - Preprocessing 완료</span>
    * Library import
    * from creatTestData import *
    * load_with_dfList(id_list)

<p style="color:#C83821; font-size:2rem; line-height: 3rem; font-weight: normal; margin-top: 10rem; margin-bottom: 10rem; text-align: left; opacity: 1;"> 1.테스트할 차량들의 DataFrame을 <strong style="color:#4263B9; opacity: 1;">아래 function</strong>으로 생성<br>
<span style="color:black; font-size:4rem; line-height: 7rem; margin-left: 20rem;">'df = loadTestSet(test_id)'</span><br>
<span style="margin-left: 70rem;">테스트 차량들의 id에 따른 동적 생성</span></p>

<p style="color:black; font-size:1.4rem; line-height: 3rem; font-weight: normal; margin-top: 10rem; margin-bottom: 10rem; text-align: right; opacity: 0.7; margin-left: 20rem;"> 
        👉🏻df생성 함수는 id_list를 인자로 받아 id갯수만큼 id명칭에 기반한 DataFrame을 생성합니다.</p>


#### 2.scale transform - Preprocessing
    * StandardSacler
#### 3.input dimension transform - Preprocessing
    * (sample, sequence, feature)
    
<p style="font-size:7rem;">☝🏻</p><br>

#### 4.test data Visualization
#### 5.load model
    * pm_1,pm25,pm10,co,co2
    * sequence 45
    * **normal model / overfitting model
#### 6.test data predict
#### 7.test result Visualization
    * inverse Sacle

# 1.test data load - Preprocessing
    * Library import : createTestData.py / from creatTestData import *
    * test_SET class
    * df = load_with_dfList(id_list)
    * test_id를 인자로 함수 실행, test_SET / info log 확인
