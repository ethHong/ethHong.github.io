---
layout: post
title:  "Review - CycleGAN"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date:   2023-05-13
last_modified_at:  2023-05-13
categories: DeepLearning GAN Tutorial
---


*본 게시글은  CycleGAN 논문과 Google 의 공식 Colab tutorial 을 참고하여 공부한 내용을 바탕으로 작성했습니다. 

## Style Transformation - CycleGAN

![image](https://user-images.githubusercontent.com/43837843/122640665-b8e19800-d13b-11eb-930e-7751ea2ac402.png)

Generative Model 을 공부하다가, CycleGAN을 이용한 흥미로운 repository 가 많음을 알게 되어 CycleGAN에 대해 공부해보게 되었습니다. GAN에서는 Generator와 Discriminator 를 이용해 '진짜같은' 데이터를 생성하는 것을 목적으로 합니다. Generator는 real data의 어떠한 분포를 학습해 더욱 진짜같은 데이터를 만들고, Discriminator는 이 이미지가 Generator로부터 생성된 것인지, 실제 데이터 분포에서 온 것인지를 구별하는 것이 목적입니다. 

**반면, CycleGAN 은 데이터를 한 도메인에서, 다른 도메인으로 'translate'하는 것을 목적으로 합니다.** 예를들어 영어를 불어로 'translate' 하는 것 처럼, 말 사진이 주어졌을 때 이미지속의 말을 얼룩말로 변형하는 것처럼 말이죠. Generation 이 된 G(x)는, X 도메인에서 Y 도메인으로 translate 된 데이터의 estimator가 됩니다. 즉 $G : X \rightarrow Y$ 이고, $\hat y = G(x)$ 가 됩니다.

그런데, CycleGAN논문에 따르면 이 방식은 단순히 generation을 위한 모델인 GAN 의 경우와 달리 문제점이 있습니다. **바로, 이렇게 translate되어 나온 데이터 G(x)가 일관성있게 X-Y 쌍으로 짝지어지리라는 보장이 없다는 것이죠.** 즉, Generator 함수를 통해 input data x가, Y 도메인의 데이터와 같은 분포를 가지는 도착점으로 매핑될 수 있는 경우의 수가 무수히 많습니다. 

조금더 풀어 설명하자면, 위의 그림속 왼쪽의 달리는 말 그림을 input으로 준다고 가정합시다. 우리가 원하는 것은 '그림속의 이 말'을 얼룩말로 바꾸는 것이죠. Y 도메인속 training dataset들은 무수히 많은 얼룩말 이미지들로 이루어져있겠죠. 그런데 G(x)로부터 생성된 가상의 얼룩말 이미지가, input data와 같은 모양을 띄고있는 얼룩말이라는 보장이 없습니다. 전혀 다른 모습을 하고있는 얼룩말의 이미지가 나올 수 있죠. 그렇다면 이것은 '말' 이미지에서 얼룩말을 '생성'한것이지, translate한 것이라 볼 수는 없습니다. 

**따라서 CycleGAN에서는 $G : X \rightarrow Y$ 와 $F : Y \rightarrow X$ 라는, 양방향으로 가는 translator를 정의해서 두 translator가 'inversible'하다는 제약조건을 걸어줍니다.** 즉, $F(G(X)) \approx x$, $G(F(Y)) \approx y$ 가 되도록 제약조건을 걸어주고, 이 역할을 바로 ***Cycle Consistency Loss***가 수행하게 됩니다. 

Google Style Transfer Colab Tutorial의 documentation에 의하면, 2015년에 나온 논문에 기반한 'Neural Style Transfer' (https://arxiv.org/abs/1508.06576) 에 비해 1000배까지도 훨씬 속도가 빠르다고 합니다. Neural Style Transfer도 나중에 깊게 따로 공부해볼 예정인데, 이 경우에는 **CNN을 사용해, Contents Image의 특성과 Style Image의 특성을 따로 학습시켜 조합하는 형식으로 학습한다고 합니다.**



## Loss Function for Cycle GAN



우선, CycleGAN의 논문을 보면 기본적으로 GAN의 Adversarial Loss (서로 반대의 목적을 가진 Discriminator 와 Generator의 Loss)를 사용합니다.

다만, $G : X \rightarrow Y$ 와 $F : Y \rightarrow X$의 두 방향에 대한 **adversarial loss**가 모두 등장합니다.

$\mathcal{L}\_{GAN}(G, D_Y, X, Y)$ 가 X에서 Y 로 갈 경우 (Discriminator on Y), $\mathcal{L}\_{GAN}(F, D_X, X, Y)$ 가 반대의 경우가 됩니다 (Discriminator on X).

여기서 **adversarial loss function**을 더 자세히 쓰자면: 

$\mathcal{L}\_{GAN}(G, D_Y, X, Y) = E\_{y~p\_{data(y)}}[log(D\_{Y}(y))] +  E\_{x~p\_{data(x)}}[log(1-D\_{Y}(G(x)))]$ 로 GAN에서 사용되는 식과 같으며, $\mathcal{L}\_{GAN}(F, D_X, X, Y)$의 경우 같은 식에 X, Y 도메인의 방향만 반대가 되겠습니다. 

**Cycle loss** 의 경우, 위에서 설명했듯 $y \rightarrow F(y) \rightarrow G(F(Y))$가 $y$와 최대한 가까운 값이 되도록 (즉, translation을 다시 reconstruct했을때 원래 input 과 최대한 오차가 적도록) 하는것이 목표입니다. CycleGAN은 $G$ 와 $F$ 두 개의 translator가 invertable하도록 만들어야 하기때문에 

 **$\mathcal{L}\_{cyc}(G, F) = E\_{x~p\_{data}(x)}[\Vert F(G(X))-x \Vert\_{1}] + E\_{y~p\_{data}(y)}[\Vert G(F(Y))-y \Vert\_{1}] $** 

로 써주게 됩니다. 

**최종적인 full objective를 보면, adversarial loss와 cycle loss를 결합한 형태가 됩니다:**

 $\mathcal{L}(G, F, D\_X, D\_Y) = \mathcal{L}\_{GAN}(G, D_Y, X, Y) + \mathcal{L}\_{GAN}(F, D_X, X, Y) + \lambda\mathcal{L}\_{cyc}(G, F)$

논문에 의하면 $\lambda$가 있는 이유는 Cycle Loss에 weight를 주어서 두 loss간의 상대적 중요도를 조정하기 위함이라고 합니다. 학습시 중요한 하나의 새로운 파라미터가 되겠죠?

결국 우리의 목적은 다음 조건을 만족하는 G와 F라는 translator를 찾는 것입니다:

 **$G^\*, F^\*  = argmin\_{G, F}  max\_{D\_{X}, D\_{Y}} \mathcal{L}(G, F, D\_X, D\_Y)$**

GAN의 경우와 같지만, 양방향성을 고려해 generator (혹은 translator), discriminator가 2쌍이라는 점, 그리고 Cycle Loss가 포함되었다는 점이 가장 큰 차이겠네요. 



## Application with Tensorflow - Colab



![image2](https://user-images.githubusercontent.com/43837843/122640726-0bbb4f80-d13c-11eb-802d-1675f780f1c6.png)

구글의 Colab 에서 제공하는 CycleGAN 튜토리얼 노트북을 기반으로 한번 실제 구현을 해보았습니다. 단, 기본적으로 제공해준 이미지의 데이터셋을 사용하지 않고 다른 이미지를 사용해봤습니다. 

**Domain X로는 CycleGAN 논문에서 사용한 풍경 이미지 중 300장을 사용했고,** transform시킬 **domain Y로는 구글에서 일본 애니메니션 풍으로 그려진 풍경 이미지 300장을 크롤링해왔습니다.** 'Anime Scenary'라는 키워드로 검색하면 위와 같은 이미지들을 얻을 수 있고, 모두 [256, 256, 3] 의 차원으로 변환해주었습니다. 

이 튜토리얼에서는 tensor flow_exmaples에서 사전 정의된 pix2pix 모델을 불러와 사용합니다. 

그리고 **generator G, generator F, discriminator X, discriminator y 를 각각 아래와 같이 정의해줍니다**

```python
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
```

그럼, 먼저 학습 이전 random noise 에서 X에서 Y로, Y에서 X로 생성한 generation을 생성해볼까요?



```python
#Define generator from X to Y, Y to X
to_B = generator_g(tf.reshape(sample_A[0].permute(0,2,3,1)[0], [1, 256, 256, 3]))
to_A = generator_f(tf.reshape(sample_B[0].permute(0,2,3,1)[0], [1, 256, 256, 3]))

plt.figure(figsize=(8, 8))
contrast = 8

imgs = [tf.reshape(sample_A[0].permute(0,2,3,1)[0], [1, 256, 256, 3]), to_B, tf.reshape(sample_B[0].permute(0,2,3,1)[0], [1, 256, 256, 3]), to_A]
title = ['Pic', 'To anime', 'Anime', 'To pic']


for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()
```

![Unknown](https://user-images.githubusercontent.com/43837843/122751761-2112b400-d2cb-11eb-8f90-a1f9be43978c.png)

참고로 이 튜토리얼의 원본은 tensorflow examples에서 제공하는 이미지세트를 로드하였지만, 저는 직접 업로드한 이미지를 torch dataloader로 로드하였기때문에 tensor의 자원이 조금 달라져서 세부 조정을 거쳐야 했습니다. 

다음으로 코드의 주요 부분들을 살펴보겠습니다. 전체 코드는 아래 링크를 첨부하고, 주요한 부분만 살펴보도록 하겠습니다. 먼저 discriminator loss와 generator loss, 그리고 cycle loss를 정의해줍니다. 

```python
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss
```

자세히 보면 identity loss라는것이 하나 추가되었습니다. 논문의 objective function 부분에는 나와있지 않지만, application 부분에 보면 **photo generation from painting 과 같은 경우 이 identity mapping을 사용해 성능을 높힐 수 있다는 내용이 나옵니다.**  Generator는 G가 X를 Y로 translate하는 역할을 해야합니다. 그렇다면 G에 Y를 input으로 주면 뭘 뱉어내야 할까요? Y를 그대로 뱉어내거나, Y와 거의 유사한 이미지를 출력해야 할 것입니다. 

여기서 Identity loss는 X를 Y 로 translate 하는 generator G에, Y를 넣었을 때 발생하는 Loss 입니다.  G(Y)와 Y, F(X)와 X간의 loss를 계산하게 됩니다. **이렇게 해서 얻는 이득은 color composition 을 원본과 유사하게 유지해줄 수 있다고 합니다.** 

처음에는 Cycle Loss가 있는데 왜 굳이 Identity mapping 이 필요할까? 라는 의문을 가졌습니다. 그런데 다시 생각해보니, Cycle Loss가 다루는 영역은 데이터의 '분포'입니다. generation 된 데이터와, 이를 다시 reconstruction 했을때 데이터들이 특정 분포를 따르게 되기때문에, loss 의 expectation이 최소화되도록 합니다. **즉 '같은 데이터의 분포에서 나왔다고 여겨진다면' (특정 스타일을 따른다고 여겨진다면) 색상과 같은 요소들을 완전 바꾸어버릴 여지가 있을 것입니다. 반 고흐의 스타일을 학습시켰다면 모든 풍경을 파랗게 바꿔버릴지도 모르죠.** 그래서 Identity mapping을 통해 원본 이미지와 color composition이 크게 벗어나지 않도록 잡아주는 것이라 합니다. ''반대로 말하면 만일 나는 극단적으로 색상과 같은 요소도 스타일에 학습시키고 싶어'' 라고 한다면 identity loss의 weight을 줄일 수도 있겠군요.

<img width="494" alt="Screen Shot 2021-06-21 at 8 43 40 PM" src="https://user-images.githubusercontent.com/43837843/122756429-5de1a980-d2d1-11eb-9586-8d8b75b1d1bf.png">

이제 이 loss들과 미리 설정한 optimizer를 가지고, 다음과 같이 training을 진행하는 함수를 정의해줍니다. X, Y에서 나오 이미지들을 각각 인풋으로 받아, G, F generator들이 각각 fake image를 생성하도록 discriminator loss, generator loss, cycle loss를 양방향으로 정의해줍니다. total_cycle_loss 에 $\lambda$를 추가해 cycle loss나 identity loss에 비중을 조절할 수 있겠지만, 튜토리얼에서는 그렇게 하지 않았습니다. 

```python
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
  
  train_G_loss(total_gen_g_loss)
  train_D_loss(disc_y_loss)
```



## Training and results 

![image3](https://user-images.githubusercontent.com/43837843/122640741-1b3a9880-d13c-11eb-836e-39427994d11c.png)

Colab Pro 의 고용량 램과 GPU를 사용하였습니다. 가장 먼저, **배치사이즈 1, 혹은 4의 extreme mini batch인 경우가 눈으로 볼때도, loss 수치상으로도 좋은 결과를 보여주었습니다.** 몇 가지 이유가 있을 것 같은데 첫번재는 전반적인 이미지 데이터셋의 퀄리티가 높지 않았고, 데이터셋이 작았으며, 이미지들의 편차가 크지 않았습니다 (전반적으로 비슷비슷 했습니다). 그래서 배치사이즈가 작았음에도 불구하고 학습이 잘 된게 아닌가 싶네요. 

![image](https://user-images.githubusercontent.com/43837843/122757494-a9488780-d2d2-11eb-8e57-5d5bd1d96014.png)

![image](https://user-images.githubusercontent.com/43837843/122757534-b1a0c280-d2d2-11eb-99e3-fe3cb018a582.png)

일단 저는 X에서 Y로의 translation 이 목적이었기때문에, generator loss 와 discriminator loss모두 X에서 Y로의 translation에 초점을 두었습니다. 수치를 보면 확실히 Discriminator loss가 낮은 수치에서 시작함을 볼 수 있는데, **이는 generator가 random noise에서부터 시작하기 때문일 것입니다.** 



![image4](https://user-images.githubusercontent.com/43837843/122640878-b7fd3600-d13c-11eb-8822-daabd6b7f857.png)

또 한가지 흥미로웠던 점은, 무한정 epoch을 키워서 트레이닝을 진행해보다 보니, 저화질 이미지들의 열화된 feature들을 학습하기 시작하더군요. 또, 전반적으로 모델이 **일본풍 애니메이션 그림체에서 많이 쓰이는 color composition을 학습했음을 알 수 있습니다. (분홍색, 채도가 높은 밝은 녹색 등)** 사실상 그림체도 조금씩 변하긴 했지만, **모네나 반 고흐의 그림을 대상으로 한 케이스에 비해 그림의 왜곡이 적어서 그런지, 색감의 왜곡이 생겨도 identity loss가 충분히 낮았던걸까? 라는 추측을 해봅니다.**  



**남겨진 몇가지 의문을 해결하기 위해서 나중에는, 1) 더 큰 데이터셋을 가지고 2) 더 높은 퀄리티의 고해상도 이미지들을 사용하여 3) Cycle Loss, Adversarial Loss, 그리고 Identity Loss의 비중을 조정하면서 학습해보아야 할 것 같습니다.** 

---

Github Repo: <https://github.com/ethHong/Cycle_GAN_tutorial_practice>

CycleGAN origianl Repo:  <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>

Colab Tutorial: <https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb#scrollTo=_xnMOsbqHz61>

CycleGAN 논문: <https://arxiv.org/abs/1703.10593>

NYU Deep Learning - Generative models, VAE: <https://atcold.github.io/pytorch-Deep-Learning/en/week08/08-3>