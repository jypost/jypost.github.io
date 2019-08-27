# 1.오호
* 이거
 * 좋구만
# ㄴㅇ륲ㄴㅇ륨ㅇㄹㅍㄴㅍ
<span > 여기에 </span>ㅁㅍㅇㄹㅍㅁㅇㄹㅍ
ㅁㅇㄹㅍㅁㄹㅍㅁㅇㄹ
ㅁㄹㅍㅁㅇㄹㅍㅁㅇㄹㅍ
ㅁㅍㅁㅇㄹㅍㅁㅇ
ㅁㅇㄹㅍㅁㅇㄹㅍㅁㅇㄹㅍ
### ㅁㄴㅍㅁㅇㅍ
[TOC]
ㅁㅇㄹㅍㅁㅇㄹㅍ
![](http://)
1. 정리하면, 
<br>
* <span style="color:#C83821;">데이터 늘이기</span>
  * imgaug 라이브러리를 사용하면 쉽고 빠르게 데이터를 늘릴수 있다.
  * 훈련과정에서 데이터 늘이기로 인해 학습 모델은 강제적으로 generalize(일반화) 된다.
  * 반론도 있다함.(이런식으로 증가된 데이터가 실제 데이터를 대체할 만한 것이 아니다)
  * 아래 데모 코드 참조
<br><br>

{% highlight python %}

   # apply augmenters in random order 순서를 랜덤하게 섞는다
    random_order=True) 

    # This should display a random set of augmentations in a window 랜덤하게 증식된 이미지를 창에 표시함
    images_aug = seq.augment_images(images)
    seq.show_grid(images[0], cols=8, rows=8)

{% endhighlight %}

_ _ _

- - -

* * *
| column | column |
|--------|--------|
|     ㄴㄴ   |    ㄴㄴㄴ    |


## 최종 결과 이미지 출력 ( 데이터가 이쪽저쪽 늘이고 줄이고 많아진것 확인 )<br>
![Imgur](https://i.imgur.com/Aa2edca.png)

==여기에 하이라이트를==
==여기에 하이라이트를==
==여기에 하이라이트를==
여기에 하이라이트를
==여기에 하이라이트를==
==여기에 하이라이트를==
==여기에 하이라이트를==
==여기에 하이라이트를==
====
	def reshape_for_lstm(data, look_back=1):
		data_X = []
		data_Y = []
		for i in range(0, len(data)-look_back):
			subset = data[i:(i+look_back)]
			for si in range(0, len(subset)):
				data_X.append(subset[si, 1:])
			data_Y.append(subset[si, 0])
		data_X, data_Y = np.array(data_X), np.array(data_Y)
		data_X = data_X.reshape((-1,look_back, data_X.shape[1]))
		return data_X, data_Y

[TOC]


## 최ㄹㅍㄴㅇㄹㅍㄹ지 출력 ( 데이터가 이쪽저쪽 늘이고 줄이고 많아진것 확인 )<br>