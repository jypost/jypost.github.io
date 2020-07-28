---
title: "About_anomaly_detection_by LSTM"
date: 2020-07-15 15:09:28 -0400
categories: DeepLearning
tags:
- anomaly detection
- LSTM
- RNN
- Generative Model
- time series anomaly detection
---

# <span style="color:black">LSTM 이상감지 모델, 이상한 점</span><br>
시작,<br>

original training data<br>
![](https://github.com/jypost/jypost.github.io/blob/master/img/LSTM_test_training_original.png?raw=true)<br>
noise added training data<br>
![](https://github.com/jypost/jypost.github.io/blob/master/img/LSTM_test_training_noise.png?raw=true)<br>

smaple : 40 test<br>
input data.shape :  (40, 60, 1)<br>
![](https://github.com/jypost/jypost.github.io/blob/master/img/LSTM_test_001.png?raw=true)<br>

smaple : 440 test<br>
input data.shape :  (440, 60, 1)<br>
![](https://github.com/jypost/jypost.github.io/blob/master/img/LSTM_test_002.png?raw=true)<br>

데이터 증식, 데이터 보강, 데이터 증강, 데이터 확장, 데이터 보완, 데이터 늘이기 등. 다양하게 번역되는 용어이다.<br>
기존 데이터를 이리저리 변형해 새로운 데이터를 만들어 기존 데이터에 덧붙이는 방식으로
기존 데이터를 보강한다는 면을 생각하면 <span style="color:#2539A6; font-size: 1.6rem;">**'데이터 늘이기'**</span>라는 말이 개념을 가장 잘 나타내는것 같다.<br>
*통계학에서는 augmentation을 확대로 해석한다고(음, 여튼 딥러닝에서는 늘이기가 쉽게 이해된다.)* <br>
데이터 늘이기는 딥러닝 데이터 분석의 초석히다. 데이터 늘이기로 인해 프로젝트가 어떻게 개선될 수 있는지를 이해하려면 각 프로젝트를 진행해 보아야 한다.<br>
왜 자신이 맡은 프로젝트에 데이터 늘이기 기능을 적요시키려 하는가? 
이미지를 예로 들어 쉽게 이해해 보자. 뒤집거나 노이즈를 추가하는 식으로 데이터를 늘임으로써, 본질적으로 **특이 특징(singular features)**을 기억하게
하거나(memorizing) 풀어 내지(keying) 않고도, 알고리즘이 이미지의 내용을 이해하게 할 수 있다.<br>

정리하면, 
<br>
* <span style="color:#C83821;">데이터 늘이기</span>
  * imgaug 라이브러리를 사용하면 쉽고 빠르게 데이터를 늘릴수 있다.
  * 훈련과정에서 데이터 늘이기로 인해 학습 모델은 강제적으로 generalize(일반화) 된다.
  * 반론도 있다함.(이런식으로 증가된 데이터가 실제 데이터를 대체할 만한 것이 아니다)
  * 아래 데모 코드 참조
<br><br>

## <span style="color:#C83821">imgaug 라이브러리를 활용하면 쉽게 data augmentation을 할 수 있다.</span><br>