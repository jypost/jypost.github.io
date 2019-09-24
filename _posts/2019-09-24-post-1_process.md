---
title: "cvalidation processing"
date: 2019-9-19 15:09:28 -0400
categories: DeepLearning
tags:
- Preprocessing
- validation model
- test model
- validation process
---

# Process
#### 1.test data load - Preprocessing
    * Library import
    * from creatTestData import *
    * load_with_dfList(id_list)
#### 2.scale transform - Preprocessing
    * StandardSacler
#### 3.input dimension transform - Preprocessing
    * (sample, sequence, feature)
#### 4.test data Visualization
#### 5.load model
    * pm_1,pm25,pm10,co,co2
    * sequence 45
    * **normal model / overfitting model
#### 6.test data predict
#### 7.test result Visualization
    * inverse Sacle
