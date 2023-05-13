---
layout: post
title:  "[KR] Overview: About Zero shot Learning and Multimodality"
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date:   2023-05-13
last_modified_at:  2023-05-13
categories: DeepLearning ZeroShotLearning Multimodality
published: true
---

# Dall-E, GPT... 기존 접근과 뭐가 다를까?

Generative AI, LLM, 빅 모델 등... Open AI 가 뛰어난 성능의 오픈소스 API 를 공개하면서 AI 가 다시한번 변곡점을 맞은 것 처럼 보입니다. 확실히 Google Trend만 보아도, ChatGPT 키워드가 부상하며 AI 키워드 역시 이전 대비 급격하게 관심도가 증가한것을 볼 수 있습니다. AI, 그리고 딥러닝에 대한 연구는 계속 다양한 분야에서 꾸준히 진행되어왔지만, End User 에게 그 발전의 속도와 정도가 피부에 와 닿게 만들어준것은 확실히 OpenAI 의 영향이 있는 것 같습니다. 

![Screenshot 2023-05-02 at 3.57.55 PM](/Users/HongSukhyun/Library/Application Support/typora-user-images/Screenshot 2023-05-02 at 3.57.55 PM.png)

GPT는 확실히 사람처럼 말을 하고, 정말 다양한 답변을 생성해냅니다. 단순히 물음에 대답할 뿐 아니라, 정보를 주었을 때 이를 효율적으로 '사람처럼' 처리하는 일에도 뛰어난 성능을 보이고 있습니다. 예를 들어, [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf) 라는 논문에서는 GPT 와 같은 'Large Language Model' 을 통해, 수학문제를 풀도록 합니다. Dall-E는 주어진 텍스트에 알맞은 이미지를 생성해냅니다. 확실히 예전에 비해 AI, 그리고 딥러닝의 방향성과 처리할 수 있는 작업의 범위가 넓어졌음을 알 수 있습니다. 최근의 Generative AI 트렌드를 주도하는 Dall-E나, GPT 같은 Large Model 들의 핵심에는 **Zero-shot (또는 Few shot), 또는 Multimodality** 가 있습니다. 저도 공부하는 입장에서, 블로그 포스트를 통해 쉽게 이해할 수 있도록 이들에 대해 다뤄보려 합니다. 이번 포스트에서는 우선 Zero shot, 그리고 Multimodality가 무엇인지, 기존 딥러닝 접근법과 무엇지 다른지를 간략하게 정리해보겠습니다. 

# Traditional Deep Learning: 학습하지 않은 데이터에 대한 결과물을 추론하기

시작하기 앞서 딥 러닝, AI 모델에 대해 간단히 짚고 넘어가겠습니다. (이 부분이 쉽게 느껴지신다면 넘어가도 됩니다.) 많은 딥러닝 모델들이 분류, 추론 등의 과제를 수행하는 방식은 보통 Data ($\x$) 와 Lable ($y$) 를 통한 학습니다. 예를들어, '말 (horse)' 이미지를 구별하는 딥러닝 모델을 만든다고 하면, 이미지 데이터 $x$가 주어졌을 때, 해당 데이터의 Label $y$ (해당 이미지가 '말' 을 포함하고 있을 확률, 또는 binary 로 포함하고 있는지 여부 등을 나타냄.)를 추론하는 모델을 만들게 됩니다. 결국

$f_{model} : \mathcal{x} \rightarrow \mathcal{y}$ 와 같이, 이미지 데이터의 도메인 $\mathcal{x} 로부터, Label의 도메인 \mathcal{y}$ 에서 추론값을 얻어내는 일종의 함수와 같은 역할을 하게 됩니다.

# Zero - shot learning: 학습하지 않은 데이터에 대한 결과물을 추론하기

