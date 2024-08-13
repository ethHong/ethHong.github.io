---
layout: post
title:  "CycleGAN - One of the brilliant style transfer models"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date: 2024-07-30
last_modified_at: 2024-07-30
categories: DeepLearning GAN Tutorial
published: true
---

* *The original post had been composed in Korean by myself, and power of AI had provided assistance for translation and revision. Contents themselves are created by  the writer, Sukhyun Hong without usage of AI tools.* 

A few years ago, before user-friendly generation AI tools came out in the world, I discovered many interesting resources related to this interesting style transfer model - 'Cycle GAN' while researching image generation. 

We have DALL-E, Midjourney, and Stable Diffusion now so image generation (mostly test to image generation) seems to be so easy today. However, before ChatGPT and the gen-AI boom, I remember GAN models being referred to as a game changer.

Since I believe the basic approach and idea of CycleGAN is very intuitive and interesting, through this post, I am trying to introduce and share what I studied several years ago about CycleGAN. 

---

## Style Transformation - CycleGAN

![image](https://user-images.githubusercontent.com/43837843/122640665-b8e19800-d13b-11eb-930e-7751ea2ac402.png)



The objective GAN Models are generating 'real like data', utilizing a generator, and discriminator. The generator tries to create more 'real-like' data by learning the distribution of real data. In contrast, the discriminator tries to identify if the data is from the generator or the actual one. 

**On the other hand, CycleGAN aims to 'translate' data from one domain to another.** For example, just as translating English into French, when given a picture of a horse, it transforms the image into a zebra. The generated output, G(x), becomes an estimator of data translated from domain X to domain Y. In other words, $G : X \rightarrow Y$, and $\hat{y} = G(x)$.

However, according to the CycleGAN paper, this method has some issues, unlike the case of GAN, which is simply a model for generation. **Specifically, there is no guarantee that the translated data G(x) will consistently match with the X-Y pair.** In other words, there are countless possible mappings where the input data x could be mapped to a point in domain Y that shares the same distribution.

To elaborate a bit more, let's assume that we provide the running horse image on the left side of the above picture as input. What we want is to transform 'this specific horse in the image' into a zebra. The training dataset in domain Y would consist of numerous zebra images. However, there is no guarantee that the virtual zebra image generated from G(x) will have the same shape as the input data. It could be a zebra with a completely different appearance. In that case, it would be considered 'generation' of a zebra image from the horse image rather than translation.

**Therefore, CycleGAN defines bidirectional translators, $G : X \rightarrow Y$ and $F : Y \rightarrow X$, and imposes a constraint that these two translators are 'invertible'.** This means that $F(G(X)) \approx x$ and $G(F(Y)) \approx y$, and this is exactly the role that ***Cycle Consistency Loss*** plays.

According to the documentation of Google Style Transfer Colab Tutorial, it is said that 'Neural Style Transfer' based on the [2015 paper](https://arxiv.org/abs/1508.06576) is up to 1000 times faster. I plan to study Neural Style Transfer more deeply later, and in this case, **it is said that CNN is used to separately learn the characteristics of the content image and the style image, and then combine them.**

## Loss Function for Cycle GAN

First of all, if you look at the CycleGAN paper, it fundamentally uses the Adversarial Loss of GAN (the loss of the Discriminator and Generator, which have opposing objectives).

However, the **adversarial loss** appears for both directions, $G : X \rightarrow Y$ and $F : Y \rightarrow X$.

> $\mathcal{L}\_{GAN}(G, D\_Y, X, Y)$

is loss function of the direction X to Y (Discriminator on Y), while

> $\mathcal{L}\_{GAN}(F, D\_X, X, Y)$

is the loss function of the other direction (Discriminator on X).

To write the **adversarial loss function** in more detail, it is :

> $\mathcal{L}\_{GAN}(G, D\_Y, X, Y) = E\_{y~p\_{data(y)}}[log(D\_{Y}(y))] +  E\_{x~p\_{data(x)}}[log(1-D\_{Y}(G(x)))]$

which is same as the formula used for GAN. $\mathcal{L}\_{GAN}(F, D\_X, X, Y)$ can also be written in same way, only changing domain direction of X and Y. 

As for the **Cycle Loss**, as explained above, the goal is to make $y \rightarrow F(y) \rightarrow G(F(y))$ as close as possible to $y$ (i.e., when the translation is reconstructed, the original input should have minimal error). CycleGAN aims to ensure that both translators, $G$ and $F$, are invertible.

So, it is written as: 

> $\mathcal{L}\_{cyc}(G, F) = E\_{x~p\_{data}(x)}[\Vert F(G(X))-x \Vert\_{1}]+ E\_{y~p\_{data}(y)}[\Vert G(F(Y))-y \Vert\_{1}]$

**The final full objective is a combination of the adversarial loss and the cycle loss:**

> $\mathcal{L}(G, F, D\_X, D\_Y) = \mathcal{L}\_{GAN}(G, D\_Y, X, Y)+ \mathcal{L}\_{GAN}(F, D\_X, X, Y) + \lambda\mathcal{L}\_{cyc}(G, F)$

According to the paper, the reason for the $\lambda$ is to give weight to the Cycle Loss, adjusting the relative importance beween the two losses. This becomes an important new parameter during training.

In the end, our goal is to find translators $G$ and $F$ that satisfy the following condition:

> $$
> G^*, F^* = \arg\min{G, F} \max{D{X}, D{Y}} \mathcal{L}(G, F, D\_X, D\_Y)
> $$


****

This is similar to the case of GANs, but with the key differences being that there are two pairs of generators (or translators) and discriminators due to the bidirectionality, and the inclusion of Cycle Loss.

## Application with Tensorflow - Colab



![image2](https://user-images.githubusercontent.com/43837843/122640726-0bbb4f80-d13c-11eb-802d-1675f780f1c6.png)

Based on the CycleGAN tutorial notebook provided by Google Colab, I attempted to implement it myself. However, I used different images instead of the default dataset provided.

**For Domain X, I used 300 landscape images from the CycleGAN paper,** and for the transformation target **Domain Y, I crawled 300 landscape images drawn in a Japanese anime style from Google.** You can find such images by searching with the keyword "Anime Scenery," and I converted all of them to a [256, 256, 3] dimension.

In this tutorial, I used a pre-defined pix2pix model from `tensorflow_examples`.

Then, **I defined the generator G, generator F, discriminator X, and discriminator Y as follows:**

```python
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
```

Let's try to create generation X to Y, Y to X which is from random noise. 

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

For reference, the original tutorial loads the image set provided by example of tensorflow, but since I loaded the images I uploaded myself using the `torch` dataloader, the tensor resources changed slightly, requiring some fine-tuning.

Next, let’s take a look at the key parts of the code. I’ll provide the full code via the link below, and we’ll focus on the main sections. First, let's define the discriminator loss, generator loss, and cycle loss.

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

If you look closely, you’ll notice that an additional component called **identity loss** has been added. Although it’s not mentioned in the objective function section of the paper, it is discussed in the application section, where it explains that **using this identity mapping can enhance performance in cases like photo generation from paintings.** The generator G is supposed to translate X to Y. So, what should G output when Y is provided as input? It should output Y itself or something very close to Y.

Here, identity loss is the loss that occurs when Y is input into the generator G, which is responsible for translating X to Y. It calculates the loss between G(Y) and Y, and between F(X) and X. **The benefit gained from this is that it helps maintain the color composition similar to the original.**

At first, I wondered why identity mapping is necessary when we already have Cycle Loss. But after reconsidering, I realized that Cycle Loss deals with the 'distribution' of the data. Since the generated data, when reconstructed, follows a certain distribution, the expectation of the loss is minimized. **In other words, if it is considered to have come from the same data distribution (or is considered to follow a specific style), there is a possibility that elements like color could be completely altered. If it learns the style of Van Gogh, it might turn all landscapes blue.** Therefore, identity mapping helps ensure that the original image's color composition does not deviate significantly. Conversely, if you want to teach the style with extreme changes in elements like color, you could reduce the weight of the identity loss.

<img width="494" alt="Screen Shot 2021-06-21 at 8 43 40 PM" src="https://user-images.githubusercontent.com/43837843/122756429-5de1a980-d2d1-11eb-9586-8d8b75b1d1bf.png">

Now, using these losses and the pre-set optimizer, we define a training function as follows. This function takes images from X and Y as inputs, and the G and F generators create fake images. We define the discriminator loss, generator loss, and cycle loss in both directions. Although we could add $\lambda$ to the `total_cycle_loss` to adjust the weight of the cycle loss or identity loss, the tutorial does not do so.

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

I used the high-memory RAM and GPU provided by Colab Pro. First and foremost, **extreme mini-batches with a batch size of 1 or 4 yielded good results both visually and in terms of loss values.** There could be a few reasons for this: first, the overall quality of the image dataset wasn't high, the dataset was small, and the images didn't vary much (they were generally quite similar). Therefore, despite the small batch size, the training seemed to go well.

![image](https://user-images.githubusercontent.com/43837843/122757494-a9488780-d2d2-11eb-8e57-5d5bd1d96014.png)

![image](https://user-images.githubusercontent.com/43837843/122757534-b1a0c280-d2d2-11eb-99e3-fe3cb018a582.png)

Since my goal was the translation from X to Y, I focused both the generator loss and discriminator loss on the translation from X to Y. Looking at the numbers, you can clearly see that the Discriminator loss starts at a lower value, **likely because the generator starts from random noise.**

![image4](https://user-images.githubusercontent.com/43837843/122640878-b7fd3600-d13c-11eb-8822-daabd6b7f857.png)

Another interesting point was that as I continued training with an increasing number of epochs, the model began to learn the degraded features of low-resolution images. Additionally, it became evident that the model had **learned the color composition commonly used in Japanese-style anime illustrations (e.g., pink, bright saturated green, etc.).** Although the art style did change slightly, **I wonder if the identity loss remained sufficiently low despite the color distortion because the images were less distorted compared to cases where the model was trained on paintings by Monet or Van Gogh?**

### For further improvement...

I may try with 1) Larger dataset, and 2) high quality (definition) images, and 3) Try to adjust ratio of Cycle Loss, Adversarial Lass, and Identity Loss.

---

[Github Repo]( <https://github.com/ethHong/Cycle_GAN_tutorial_practice>)

[CycleGAN origianl Repo]( <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>)

[Colab Tutorial](<https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb#scrollTo=_xnMOsbqHz61>)

[CycleGAN Paper]( <https://arxiv.org/abs/1703.10593>)

[NYU Deep Learning - Generative models, VAE](<https://atcold.github.io/pytorch-Deep-Learning/en/week08/08-3>)