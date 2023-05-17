---
layout: post
title:  "[KR] Overview on Zero Shot Learning and Multimodality"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date:   2023-05-13
last_modified_at:  2023-05-13
categories: 누구나가볍게읽기 DeepLearning ZeroShotLearning Multimodality
published: true
---

*본 게시글은, [A Review of Generalized Zero-Shot Learning Methods](https://arxiv.org/abs/2011.08641) 논문의 전반부 내용을 바탕으로 공부한 내용을 함께 다룹니다. 

# What is Zero shot learning, and why it's awsome?

Generative AI, LLM, 빅 모델 등... Open AI 가 뛰어난 성능의 오픈소스 API 를 공개하면서 AI 가 다시한번 변곡점을 맞은 것 처럼 보입니다. 확실히 Google Trend만 보아도, ChatGPT 키워드가 부상하며 AI 키워드 역시 이전 대비 급격하게 관심도가 증가한것을 볼 수 있습니다. AI, 그리고 딥러닝에 대한 연구는 계속 다양한 분야에서 꾸준히 진행되어왔지만, End User 에게 그 발전의 속도와 정도가 피부에 와 닿게 만들어준것은 확실히 OpenAI 의 영향이 있는 것 같습니다. 

<img width="979" alt="Screenshot 2023-05-02 at 3 57 55 PM" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/1365b468-4c15-4093-b499-b61935f603d4">

GPT는 확실히 사람처럼 말을 하고, 정말 다양한 답변을 생성해냅니다. 단순히 물음에 대답할 뿐 아니라, 정보를 주었을 때 이를 효율적으로 '사람처럼' 처리하는 일에도 뛰어난 성능을 보이고 있습니다. 예를 들어, [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf) 라는 논문에서는 GPT 와 같은 'Large Language Model' 을 통해, 수학문제를 풀도록 합니다. Dall-E는 주어진 텍스트에 알맞은 이미지를 생성해냅니다. 확실히 예전에 비해 AI, 그리고 딥러닝의 방향성과 처리할 수 있는 작업의 범위가 넓어졌음을 알 수 있습니다. 최근의 Generative AI 트렌드를 주도하는 Dall-E나, GPT 같은 Large Model 들의 핵심에는 'semantic representation'을 학습하는 방향의 최신 연구 트렌드가 있습니다. 

그래서 이런 최신 Generative AI 모델들을 공부하기 앞서, **Zero-shot (또는 Few shot), 또는 Multimodality** 와 같은 트렌드에 대해 짚고 넘어가보려 합니다. 이번 포스트에서는 우선 Zero shot이 무엇인지, 기존 딥러닝 접근법과 이런 접근이 무엇지 다른지를 간략하게 정리해보겠습니다. 

# Traditional Deep Learning

## **학습한 데이터에 대한 결과물을 추론하기**

### **Model 은 어떻게 작동하는가?**

시작하기 앞서 딥 러닝, AI 모델에 대해 간단히 짚고 넘어가겠습니다. (이 부분이 쉽게 느껴지신다면 넘어가도 됩니다.) 많은 딥러닝 모델들이 분류, 추론 등의 과제를 수행하는 방식은 보통 Data ($x$) 와 Lable ($y$) 를 통한 학습니다. 예를들어, '말 (horse)' 이미지를 구별하는 딥러닝 모델을 만든다고 하면, 이미지 데이터 $x$가 주어졌을 때, 해당 데이터의 Label $y$ (해당 이미지가 '말' 을 포함하고 있을 확률, 또는 binary 로 포함하고 있는지 여부 등을 나타냄.)를 추론하는 모델을 만들게 됩니다. 결국 우리가 '모델' 이라고 부르는 것은, $f_{model} : \mathcal{x} \rightarrow \mathcal{y}$ 와 같이, 이미지 데이터의 도메인 $\mathcal{x}$ 로부터, Label의 도메인 $\mathcal{y}$ 에서 추론값을 얻어내는 일종의 함수와 같은 역할을 하게 됩니다.  

가장 Simple 한 형태의 '함수' 가 무엇이 있을까요? $y = ax$ 형태의 일차함수, 또는 선형 함수가 있을 것입니다. Input 데이터 (변수) $x$에, 얼만큼의 'Weight'이 곱해져 $y$라는 output이 나오게 되는 데이터적인 Trend가 있는지 선형 관계가 있는 데이터에 대해 모델링하는 가장 대표적인 방법으로 '선형 회귀'가 있습니다. 이런 선형적인 접근은 단순히 1차원 회귀 뿐 아니라, 벡터와 행렬 연산을 통해 output 값에 영향을 미치는 input 변수가 여러 개인 다차원 회귀가 가능합니다. 

<center><img width="570" alt="Screenshot 2023-05-15 at 1 18 27 PM" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/c326233f-144c-44d9-b601-3462304e6aac"> </center>

그런데 이런 선형적인 접근 방법에는 근본적인 문제가 있습니다. 바로 실제 세상의 수많은 데이터들은 '선형적이지 않다'는 점 입니다. 예를들어, 이미지를 인식하는 'Computer Vision'분야를 생각해봅시다. 우리는 사진을 보고 고양이인지, 강아지인지 등을 구별하기 위해 귀는 어떻게 생겼고, 코는 어떻게 생겼으며, 색깔은 어떠한지 등, 각 부분 부분에 대한 정보를 인지해서 판단합니다. 이런 복잡한 판단을 하기에 선형 모델은 너무 단순합니다. 

### **신경망 - 중요한 정보를 남기고 정보의 차원을 축소하기**

<center><img width="50%" alt="Untitled-8" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/03ffe037-47d6-4640-b90d-b62276f7b024"></center>

신경망의 등장은 더 고차원적인 데이터를 처리할 수 있도록 딥러닝 모델들의 발전을 이뤄냈습니다. 수많은 모델들이 구현 방식, 형태는 다르지만 '신경망'에 기반하고 있습니다. 예를들어 '고양이'를 구별해내기 위해 더 중요한 정보와, 덜 중요한 정보가 있다고 가정해봅시다. 뾰족한 귀가 고양이를 판단하는데 가장 중요한 정보라고 생각해보죠. 색, 크기, 눈의 모양, 귀 모양과 같은 다양한 데이터들 중 '귀의 모양'에 대한 정보에 가장 큰 가중치를 부여해 판단하는 모델을 만들어야 할 것입니다. 만일 모든 동물이 '눈 모양'은 비슷하게 생겨서 크게 중요한 변수가 되지 못한다면, '눈 모양'에 대한 데이터는 적은 가중치로 학습에 반영해도 될 것이구요. 이것이 바로 신경망을 통한 차원 축소의 기본 원리입니다. 

아래와 같은 신경망 구조에서, Input data $X = [x_1, x_2 ... x_n]$는 vector화되어 신경망을 통과하게 됩니다.  벡터 데이터는 Weight vector $W = [w_1, w_2...]$ 이 곱해져 다음 Hidden layer로 넘어가게 됩니다. Weigh을 곱해 신경망을 통과한다는것은, **여러 신경망을 거치며 목표한 학습 task 에 중요한 정보를 남기고, 중요하지 않는 정보는 축소하는쪽으로 데이터의 차원이 축소됨을 의미합니다.** 가장 분류 성능이 좋은 Weigt들의 조합을 학습시키는것이 바로 '신경망 모델의 학습'이고, 잘 학습된 모델은 학습에 사용되지 않는 데이터를 입력하더라고  분류작업을 잘 해내기를 기대합니다.

**(신경망이 기본적으로 각광받는 이유, 차원 축소와 Representation learning에 대해서는 별도 포스트를 통해 조금 더 구체적으로 알아보겠습니다.)**

<center><img width="50%" alt="Screenshot 2023-05-13 at 7 34 43 PM" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/9765733d-c185-4759-9e01-d10e28b6c874"></center>

물론 알맞은 모델을 학습시키기 위해서는, 많은 양의 데이터가 필요합니다. 많은 모델들이 신경망 구조를 변형하고, 원하는 결과물을 내도록 학습하기 위한 다량의 데이터를 정제하고, 수집해왔으며 한가지 목적으로 학습된 모델이 다른 작업을 수행하기 위해서는, 또 다른 데이터로 학습하고, 튜닝하는 과정이 필요하죠. 

### **Problem is here...** 

그런데, Zero Shot Learning, 그리고 Generalized Zero Shot Learning에 대해 조명한 논문인 [A Review of Generalized Zero-Shot Learning Methods](https://arxiv.org/abs/2011.08641) 에서는 이러한 전통적인 접근법에 대한 한계점을 지적합니다. 

우선, 학습하고자하는 **목표와 부합하도록, 모든 'Class' (Label)들에 대한 충분한 라벨링 데이터를 얻는것은 쉽지 않습니다.** 위에서 예시로 든 '고양이'를 구별하는 모델을 학습하기 위해서는, 품질이 좋은 고양이의 이미지들을 아주 많이 모아야 합니다. 그런데 만일 '특정 종류의' 고양이를 구별하기 위해 모델을 활용하고자 한다면 특정 품종의 고양이 이미지 데이터를 새로 구축해야 합니다. 실제 현업에서 활용할 딥러닝 모델을 학습하기 위해 가장 큰 걸림돌이 되는 부분중 하나가 '적합한 품질의 데이터의 부족'입니다. 이에 따라 실제로 수많은 국 내/외 기업들이 학습용 데이터를 수집, 처리하고 이를 납품하기도 합니다. AWS 는 [SageMaker Ground Truth](https://aws.amazon.com/sagemaker/data-labeling/?sagemaker-data-wrangler-whats-new.sort-by=item.additionalFields.postDateTime&sagemaker-data-wrangler-whats-new.sort-order=desc)라는, 데이터 라벨링 서비스를 제공합니다. 국내에도 크라우드소싱 기반으로 데이터 라벨링을 의뢰받아 진행하는 플랫폼들이 많이 생겨났고, 해외에도 [Toloka.ai](https://toloka.ai/tolokers/)와 같은 학습용 데이터 구축 플랫폼이 있습니다. 

그만큼 **노동력과 비용**이 많이 들고, **특정 도메인에 대한 지식** (예: 형태소 데이터, 번역 데이터) 이 필요하며, 현실에서 충분한 양의 데이터를 구하기 어려운 분야들도 있습니다. 이를 해소하기 위해 적은 양의 Sample data를 통해 학습하는 Few shot, One shot과 같은 접근법 (나중에 따로 다뤄보도록 하겠습니다.) 들도 있지만 이들 역시 'Unseen Class' - 즉 모델이 본 적 없는 데이터는 구별할 수 없다는 한계점이 있습니다.

즉, 모델은 '본 적 있는 데이터'만 구별할 수 있기에, 모델에게 점점 더 많은, 다양한 데이터를 줘야만 하는 상황인 것이죠. 

# Zero - shot learning

## **학습하지 않은 데이터에 대한 결과물 추론하기**

이를 해소하기 위해 나온 Zero-shot learning 은, 더욱 사람과 같은 접근을 통해 모델이 직접 본 적 없는 데이터에 대한 추론도 가능하게 합니다. 예를들어, 사람은, '얼룩말'을 본 적이 없더라도 '말'을 본 적이 있다면, '줄무늬가 있는 말이다'라는 정보를 통해 얼룩말을 상상할 수 있습니다. 이와 같이 Zero shot learning은, 모델이 본 정보 (Seen Class)에 대한 지식을 전이 (transfer) 하여, 보지 못한 데이터 (Unseen class)를 예측합니다. 이를 가능하게 하기 위해, Zero Shot Learning (ZSL) 에선 **Seen & Unseen class의 'Name'을, high dimensional vector에 embedding**한다고 합니다. 이게 무슨 의미일까요?

> The semantic information embeds the names of both seen and unseen classes in high- dimensional vectors. Semantic information can be manually defined attribute vectors [14], automatically extracted word vectors [15], context-based embedding [16], or their combinations [17], [18]. In other words, ZSL uses seman- tic information to bridge the gap between the seen and unseen classes.

Seen class이든 Unseen class이든, 이들은 'semantic interpretation'을 가지고 있습니다. 얼룩말을 0, 일반 말을 1이라고 'Label'을 붙인다면, 0과 1 자체는 숫자일 뿐이지만 이들이 실제로 나타내는 것 (representation)은  훨씬 고차원적인 정보를 담고 있습니다. 사람이 붙인 1 이라는 Label은 덩치가 크고, 빠르게 달릴 수 있으며 일반적으로 검은색, 적갈색 또는 흰색을 띄는 포유류를 가르킵니다. 0이 가르키는 '얼룩말'은 말과 거의 동일하게 생겼고, 행동하지만 줄무늬가 있는 동물의 의미합니다. 이러한 고차원적인 의미정보를 'Semantic information' 이라고 합니다. 이런 고차원적인 의미정보를 나타낼 수 있는 벡터 공간을 학습시키는것이 바로 embedding입니다. 즉, 위 논문에서 말하는 context based embedding, word vector 등은 단어 (또는 단어가 의미하는 것) 의 정보를 담고있는 embedding space를 학습해 활용합니다. 

기존의 학습 방식이라면, 말과 유사한 이미지는 '0', 얼룩말과 유사한 이미지는 '1'이라는 label 을 구별하는 방식으로만 학습했을 테지만, 이렇게 semantic information 을 고려한 방식은 어떤 정보가 더 '말' 이 의미하는것과 가까우며, 어떤 정보가 더 '얼룩말' 이 의미하는것과 가까운지 알 수 있습니다. (Embedding Space에 대한 더 구체적인 내용은, 차후 다른 게시물을 통해 깊게 파보겠습니다.)

> This learning paradigm can be compared to a human when recognizing a new object by measuring the likelihoods between its descriptions and the previously learned notions [19].

사람은 기존에 학습했던 내용들과, 이미 학습한 내용들의 연관성을 토대로 본 적 없는 (배운적 없는) 내용에 대해서도 추론, 상상, 분류할 수 있습니다. '말', 과 '줄무늬'가 무엇인지 알고있다면 ''줄무늬가 있는 말' 의 이미지 역시 상상할 수 있겠죠. 이렇게 Semantic information를 통해, 정확히 pre-defined class의 분류를 수행하는것이 아닌, **의미적 정보를 활용해 직접 보지 못한 데이터에 대한 처리를 하도록 하는것이 Zero shot learning의 핵심입니다.**

### Generalize Zero shot 

위 논문에서는 Zero shot 만큼, Generalize Zero Shot 에도 주목하고 있습니다. 일반적인 ZSL 기법들은 'Unseen class'의 예측을 목적으로 하다보니 모델의 검증 과정에서 Unseen data 를 test set으로 사용하고, **오히려 unseen class에 오버피팅 (overfitting)되는 경향이 있다고 합니다.** 그러나 실제로 모델을 더 잘 활용하기 위해서는 unseen class 만큼, seen class 에 대해서도 예측을 잘 해내야 합니다. **실제로 사람이 사물이나 지식을 인식하는 방식이 바로 이렇기 때문이죠.**

따라서, Human regocnition 을 더 잘 모사 (imitate)하기 위해, 약 2016년도부터 Generalized ZSL (GZSL) 테크닉에 대한 연구가 늘어났습니다. Zero shot setting에서 퍼포먼스가 좋은 테크닉들이, Generalized zero shot setting에서는 오버피팅으로 인해 성능이 낮았다고 합니다.

이를 해결하기 위해 Generalized Zero Shot Learning은, 동일한 학습 방식을 차용하지만, '검증 방식'에서 차별점을 둡니다. Seen class, unseen class 모두에 대해 예측 검증을 진행하는것이죠. 

### Formulation

 Zero shot learning 의 문제를 다음과 같이 Formulate 하는데, 수식을 구체적으로 이해하기 다소 추상적이지만, 아래 식이 무엇을 '의미'하는지 하나씩 뜯어보겠습니다.

> $ S = {\{(x_i^s, a_i^s, y_i^s)_{i=1}^{N_s}} \mid x_i^s \in X^s, a_i^s i\in A^s, y_i^s \in Y^s\} $

Seen data의 Set을 의미합니다. Seen data $x_i^s$ 가 주어졌을 때의 Set이며, $X$, $A$, $Y$에 속하는 변수들은 각각 'Input data', 'semantic representation', 'label (class)'들의 데이터가 속한 Set입니다. 

> $ U = {\{(x_j^u, a_j^u, y_j^u)_{j=1}^{N_u}} \mid x_j^s \in X^s, a_j^s j\in A^s, y_j^s \in Y^s\} $

마찬가지 형태의 데이터 Set이나,  $U$에 속하는 데이터는 'Unseen data'의 Set 입니다. 

여기서, 
$$
{x_i}^s,{x_j}^u \in R^D
$$
라는 정의가 나옵니다. 즉, Seen data, unseen data 모두 Input data $x$는 $D$차원의 벡터공간에 있는 데이터임을 상정합니다. 이것이 의미하는 바는 즉, Unseen data 역시 Seen data와 같은 차원 벡터공간에 있는 데이터임을 의미합니다. 이미지를 Deep Learning 신경망을 통과시켜, '말' 이라는 동물의 특성을 잘 나타내는 방식으로 데이터를 차원 축소하는 pre-trained model 이 있다고 가정합니다. 이미 모델이 본 말 이미지를 통해, 보지 못한 (예컨데 얼룩말) 형태의 말 이미지를 예측, 또는 생성하기 위해서는 unseen data input 역시 데이터의 차원이 같아야겠죠. 

> ${x_i}^s,{x_j}^u \in R^D$ indicate the D-dimensional images (visual features) in the feature space $mathcal{X}$ that can be obtained using a pre-trained deep learning model such as ResNet [28], VGG-19 [29], GoogLeNet [30].

ResNet, GoogleNet 등으로 '말' 이라는 동물의 특성을 잘 나타내는 모델 학습을 진행했다면, 이 모델은 '말' 이라는 특성이 잘 드러나는 Feature Space ($\mathcal{x}$)을 그려낼 것입니다. 즉, $D$ 차원 데이터인 seen, unseen data $x$는 모두 Feature Space ($\mathcal{x}$)에 있다고 가정합니다. Feature Space ($\mathcal{x}$)에 어떠한 이미지 데이터를 투사하게 되면, **'말'과 관련된 신경망이 학습한 특성**을 을 잘 나타내는 방향으로 데이터가 투사되겠죠.

마찬가지고 semantic representation 들을 나타내는 space $\mathcal{A}$, 'label'들을 나타내는 space $\mathcal{Y}$에 seen & unseen data들의 semantic representation과 label들인 $a$, $y$가 포함되어있다고 가정합니다. 앞서 말한 Feature space $\mathcal{x}$가, 신경망을 통해 특정 데이터의 representation 을 변형시켜 feature를 나타내는 공간이라면, semantic representation space는 word embedding 등을 통해 만들어진 space, $\mathcal{y}$는 feature와, semantic representation 정보를 통해 최종적으로 얻어지는 class들이 있는 공간입니다. 

만약 text를 통해 이미지를 생성해내는 모델이 있다고 가정합니다. Input text 들의 feature를 목적에 맞게 변형시키는 신경망 모델은, '텍스트' 라는 데이터의 특성을 나타내는 공간에 데이터를 투사합니다. Embedding 을 통해 학습된 텍스트의 embedding space는 semantic representation을 담고 있습니다. 그리고, 텍스트를 통해 '생성되기를 기대하는' 모든 이미지의 데이터들이 담긴 공간이 바로 $\mathcal{y}$라고 보면 될 것 같습니다. 
$$
f_{GZSL} : \mathcal{X} \rightarrow \mathcal{Y}
$$
최종적으로, GZSL 은 Seen & unseen data 전체로 구성된 데이터의 공간에서,  semantic information 을 활용해 Label 을 예측하는 모델이라고 볼 수 있습니다. 

이번 게시글에서는 이렇게 간단히 개요를 알아보고, 별도 게시글에서 representation learning, 그리고 Zero shot learning / Generalized Zero shot learning에 대한 더 자세한 부분을 공부해보도록 하겠습니다. 

---

# 참고논문

[A Review of Generalized Zero-Shot Learning Methods, Pourpanah et al](https://arxiv.org/abs/2011.08641) 

[Language Models are Few-Shot Learners, Brown et al](https://arxiv.org/abs/2005.14165) 

(요 두개 논문은 더 깊게 공부해서, 다음번에 따로 다뤄보도록 하겠습니다.)

