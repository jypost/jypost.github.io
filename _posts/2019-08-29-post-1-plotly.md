---
title: "오프라인 플로팅"
date: 2019-8-29 15:09:28 -0400
categories: Deep Learning
tags:
- Data Visualizing
- Basic
- chart
---

# plotly.offline.plot()
* 로컬에 그리고자하는 그래프를 웹브라우저에서 열리는 HTML을 만들고 독립 실행 형태로 만든다.
* Ploty 버전 1.9.4 이상이 필요


{% highlight python %}
import plotly
plotly.__version__
{% endhighlight %}

## 샘플코드 
{% highlight python %}

# 그래프로 그리고자하는 값을 각각의 trace로 만든다.
# x에 범위를 지정하고, y에 값을 지정하면 됨

trace1 = go.Scatter(x=np.arange(start, end, step), y=[ndarray], mode='lines', name='train')
trace2 = go.Scatter(x=np.arange(start, end, step), y=[ndarray], mode='lines', name='test')
trace3 = go.Scatter(x=np.arange(1, len(train_set), 1), y=train_set, mode='lines', name='original')

# 만든 각각의 trace를 묶는다.
data = [trace, trace2, trace3]

py.offline.plot(data)
{% endhighlight %}

위의 코드를 실행하면 
해당 파일의 경로 폴더에 temp-plot.html 이 생성되어 있는것을 확인 할 수 있다.
