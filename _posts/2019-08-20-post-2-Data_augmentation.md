---
title: "데이터가 부족할땐, Data_augmentation"
date: 2019-8-20 20:09:28 -0400
categories: Deep Learning
tags:
- GAN
- Generative model
- Data augmentation
- Data preprocessing
---

# <span style="color:black">Data_augmentation</span><br>
번역해보면,<br>
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

{% highlight python %}

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

#여기 보이는 시드 값을 무작위 시드 값으로 변경할 수 있다.
ia.seed(1)

# #100개 이미지로 구성된 배치(batch) Example 
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(100)],
    dtype=np.uint8
)

# 서로 다른 확대방식들을 지정해서, 변환 함수를 생성함.
seq = iaa.Sequential([
    # Horizontal Flips, 수평기준 뒤집기
    iaa.Fliplr(0.5), 

    # Random Crops, 임의로 그림 크롭 함
    iaa.Crop(percent=(0, 0.1)), 

    # Gaussian blur for 50% of the images 이미지 중 50%에 가우시안 블러 먹임
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image. 각 이미지에 콘트라스트(대비) 강하게 또는 약하게 한다
    iaa.ContrastNormalization((0.75, 1.5)),

    # Add gaussian noise. 가우시안 노이즈 추가
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),

    # Make some images brighter and some darker. 일부이미지들을 밝거나 어둡게 한다
    iaa.Multiply((0.8, 1.2), per_channel=0.2),

    # Apply affine transformations to each image. 각 이미지에 아핀변환 적용함 (스케일,좌우상하위치이동,각도,줄이기늘이기)
    iaa.Affine(
        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
        translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        rotate=(-10, 10),
        shear=(-10, 10)
    )
], 
# apply augmenters in random order 순서를 랜덤하게 섞는다
random_order=True) 

# This should display a random set of augmentations in a window 랜덤하게 증식된 이미지를 창에 표시함
images_aug = seq.augment_images(images)
seq.show_grid(images[0], cols=8, rows=8)

{% endhighlight %}

## 최종 결과 이미지 출력 ( 데이터가 이쪽저쪽 늘이고 줄이고 많아진것 확인 )<br>
![Imgur](https://i.imgur.com/Aa2edca.png)
