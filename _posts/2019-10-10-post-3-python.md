---
title: "python 정리"
date: 2019-10-10 15:09:28 -0400
categories: DeepLearning
tags:
- Preprocessing
- Python
- Tips
- 주요 문법
- 파이썬
---

# python 자주쓰는 문법 정리
## 반복문 한줄로

```python
#1열 for문 예시
num = list(range(3))
dd = [e +1 for e in num] 
dd #0~2가 아닌, 1~3이 찍힌다. 각 원소에 1을 + 했음
```
 - 결과 : [1, 2, 3]
