---
layout: post
title:  "What is Zero shot learning, and why it's awsome?"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date:   2024-07-11
last_modified_at:  2023-07-11
categories: DeepLearning ZeroShotLearning Multimodality
published: true
---

* This post refers to the initial part of a paper - [A Review of Generalized Zero-Shot Learning Methods](https://arxiv.org/abs/2011.08641) and tries to review what I studies.
* *The original post had been composed in Korean by myself, and power of AI had provided assistance for translation and revision. Contents themselves are created by  the writer, Sukhyun Hong without usage of AI tools.* 

# What is Zero shot learning, and why it's awsome?

Generative AI, LLM, big models, and more... With OpenAI releasing high-performance open-source APIs, it seems that AI has once again reached an inflection point. A quick look at Google Trends shows that as the keyword "ChatGPT" has risen in popularity, interest in AI has also sharply increased compared to before. Although research in AI and deep learning has been consistently progressing in various fields, it's clear that OpenAI has played a significant role in making the speed and extent of this progress tangible for the end user.

<img width="1045" alt="Screenshot 2024-08-11 at 2 55 10 PM" src="https://github.com/user-attachments/assets/f8e52c4d-b620-40b3-96c3-d86ec757ba68">

GPT indeed speaks like a human and generates incredibly diverse responses. It not only answers questions but also excels at processing information efficiently and 'like a human.' For example, the paper [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf) demonstrates how a 'Large Language Model' like GPT can solve math problems. DALL-E generates images that match the given text. Clearly, the direction of AI and deep learning and the range of tasks they can handle have expanded significantly compared to the past. At the core of recent Generative AI trends, led by models like DALL-E and GPT, is the latest research trend focused on learning 'semantic representation.'

Before diving into these cutting-edge Generative AI models, I want to touch on trends like **Zero-shot (or Few-shot) learning and Multimodality**. In this post, I will start by briefly explaining what Zero-shot learning is and how it differs from traditional deep learning approaches.

# Traditional Deep Learning

## **Inferrence of outcome from trained data**

### How does traditional models work?

Before we begin, let's briefly go over deep learning and AI models. (If this part feels easy to you, feel free to skip it.) Many deep learning models perform tasks such as classification and inference through learning from data ($x$) and labels ($y$). For example, if you're creating a deep learning model to distinguish between images of 'horses,' the model will learn to predict the label $y$ (which indicates the probability that the given image contains a 'horse,' or whether it contains a horse in binary terms) based on the input image data $x$. In essence, what we call a 'model' functions as a type of function, $f_{model} : \mathcal{x} \rightarrow \mathcal{y}$, which takes data from the domain $\mathcal{x}$ (image data) and outputs predictions in the domain $\mathcal{y}$ (labels).

What is the simplest form of a 'function'? It could be a linear function, such as $y = ax$. In linear regression, one of the most basic modeling methods for linear relationships, we model how much 'weight' is multiplied by the input data (variable) $x$ to produce the output $y$. This approach isn't limited to just one-dimensional regression; it can also be applied to multidimensional regression, where multiple input variables affect the output value, through vector and matrix operations.

<center><img width="570" alt="Screenshot 2023-05-15 at 1 18 27 PM" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/c326233f-144c-44d9-b601-3462304e6aac"> </center>

### Problem is... world is not 'Linear!'

However, **there's a fundamental problem with this linear approach:** **most real-world data is not 'linear.'** For example, let's consider the field of 'Computer Vision,' which deals with recognizing images. When we look at a photo to determine whether it's a cat or a dog, we consider various details such as the shape of the ears, the nose, and the color. A linear model is too simplistic for making such complex judgments.

### **Neural Networks -  Reducing the dimensionality of data while retaining important information.**

<center><img width="50%" alt="Untitled-8" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/03ffe037-47d6-4640-b90d-b62276f7b024"></center>

The advent of neural networks has enabled the development of deep learning models capable of handling higher-dimensional data. Although many models differ in their implementation and structure, they are all based on 'neural networks.' For example, let's assume that some information is more important than others for distinguishing a 'cat.' Suppose the sharpness of the ears is considered the most important feature for identifying a cat. In that case, a model should be designed to assign the highest weight to the information about ear shape among various data such as color, size, eye shape, and ear shape. If the eye shape is similar across most animals and doesn't serve as a significant distinguishing factor, then the data about eye shape could be assigned a lower weight in the learning process. This is the basic principle of dimensionality reduction through neural networks.

In a neural network structure like the one below, the input data $X = [x_1, x_2 ... x_n]$ is vectorized and passed through the neural network. The vector data is multiplied by a weight vector $W = [w_1, w_2...]$ and then passed to the next hidden layer. Multiplying by weights and passing through the neural network means that **as data passes through multiple layers, the dimensionality of the data is reduced, retaining the information important to the learning task and reducing less important information.** Training a neural network model involves learning the combination of weights that best enhances classification performance, and a well-trained model is expected to perform well on classification tasks even when fed with data not used during training.

**(We will explore the reasons why neural networks are highly regarded, as well as dimensionality reduction and representation learning, in more detail in a separate post.)**

<center><img width="50%" alt="Screenshot 2023-05-13 at 7 34 43 PM" src="https://github.com/ethHong/ethHong.github.io/assets/43837843/9765733d-c185-4759-9e01-d10e28b6c874"></center>

Of course, a large amount of data is necessary to train an appropriate model. Many models have modified neural network structures and refined and collected large amounts of data to train the model to produce the desired output. If a model trained for one specific task needs to perform another task, it must undergo further training and tuning with different data.

### **We till got limitations from Neural Network!*

However, the paper [A Review of Generalized Zero-Shot Learning Methods](https://arxiv.org/abs/2011.08641), which highlights Zero-Shot Learning and Generalized Zero-Shot Learning, points out the limitations of these traditional approaches.

First of all, **it is not easy to obtain sufficient labeled data for all 'classes' (labels) that align with the learning objective.** For example, to train a model that distinguishes 'cats,' you need to collect a large number of high-quality images of cats. But if you want to use the model to distinguish a 'specific breed' of cat, you would need to gather new image data specific to that breed. One of the biggest obstacles to training deep learning models for practical use is the 'lack of suitable quality data.' As a result, many companies worldwide are engaged in collecting, processing, and delivering training data. AWS offers a data labeling service called [SageMaker Ground Truth](https://aws.amazon.com/sagemaker/data-labeling/?sagemaker-data-wrangler-whats-new.sort-by=item.additionalFields.postDateTime&sagemaker-data-wrangler-whats-new.sort-order=desc). In Korea, numerous platforms have emerged that carry out data labeling through crowdsourcing, and internationally, there are platforms like [Toloka.ai](https://toloka.ai/tolokers/) for building training data.

This process requires a significant amount of **labor and cost** and **domain-specific knowledge** (e.g., morphological data, translation data). Additionally, some fields find it challenging to gather sufficient data in practice. To address this, approaches like Few-Shot and One-Shot learning, which train models with small amounts of sample data (topics I'll cover separately later), have been developed. However, these methods also have the limitation that they cannot distinguish 'unseen classes'—that is, data the model has never encountered before.

In other words, the model can only distinguish data it has 'seen' before, which creates a situation where we need to provide the model with more and more diverse data.

# Zero - shot learning

## **Inferring outcomes for unseen (untrained) data**

Zero-shot learning, developed to address this issue, allows models to infer outcomes for data they haven't directly encountered, using a more human-like approach. For example, a person who has never seen a 'zebra' can imagine one by knowing it's a 'horse with stripes,' based on prior knowledge of what a horse looks like. Similarly, Zero-Shot Learning (ZSL) enables a model to transfer knowledge from the data it has seen (Seen Class) to predict data it hasn't seen (Unseen Class). To make this possible, Zero-Shot Learning involves **embedding the 'names' of both Seen and Unseen classes into a high-dimensional vector.** What does this mean?

> The semantic information embeds the names of both seen and unseen classes in high- dimensional vectors. Semantic information can be manually defined attribute vectors [14], automatically extracted word vectors [15], context-based embedding [16], or their combinations [17], [18]. In other words, ZSL uses seman- tic information to bridge the gap between the seen and unseen classes.

Whether it's a seen class or an unseen class, they all have 'semantic interpretation.' If we label a zebra as 0 and a regular horse as 1, the numbers 0 and 1 are just digits, but what they actually represent is much more complex. The label 1, assigned by a person, refers to a large mammal that can run fast and typically has black, chestnut, or white coloring. The label 0, representing a 'zebra,' refers to an animal that looks and behaves almost identically to a horse but has stripes. This high-dimensional information is called 'semantic information.' Embedding is the process of training a vector space that can represent this kind of high-dimensional semantic information. In other words, the context-based embedding, word vectors, etc., mentioned in the paper refer to training and utilizing an embedding space that contains the information of a word (or what the word represents).

In traditional learning methods, a model would learn to distinguish between images similar to a horse as '0' and those similar to a zebra as '1.' However, with this approach that considers semantic information, the model can understand which information is closer to what 'horse' means and which is closer to what 'zebra' means. (I'll delve deeper into the specifics of Embedding Space in a future post.)

> This learning paradigm can be compared to a human when recognizing a new object by measuring the likelihoods between its descriptions and the previously learned notions [19].

Humans can infer, imagine, and classify new (unlearned) content based on previously learned information and the relationships between what they have already learned. If someone knows what a 'horse' and 'stripes' are, they can imagine an image of a 'striped horse.' The core of Zero-Shot Learning is to use **semantic information to process data that has not been directly seen, rather than performing classification based solely on predefined classes.**

### Generalize Zero shot 

The paper highlights Generalized Zero-Shot Learning (GZSL) as much as Zero-Shot Learning (ZSL). Traditional ZSL methods often focus on predicting 'unseen classes,' leading to the use of unseen data as the test set during the model validation process. This can result in **overfitting to the unseen classes.** However, to make a model more useful, it needs to perform well on seen classes as well as unseen ones. **This approach mirrors how humans recognize objects and knowledge.**

To better imitate human recognition, research on Generalized ZSL (GZSL) techniques has increased since around 2016. Techniques that perform well in a Zero-Shot setting often show lower performance in a Generalized Zero-Shot setting due to overfitting.

To address this, Generalized Zero-Shot Learning adopts the same learning approach but differentiates itself in the 'validation method.' It validates predictions on both seen and unseen classes.

### Formulation

Zero-Shot Learning poses a challenge that can be formulated as follows. The mathematical formulation might seem abstract and difficult to understand at first, but let's break down what each part of the following equation 'means.'

> $ S = {\{(x_i^s, a_i^s, y_i^s)_{i=1}^{N_s}} \mid x_i^s \in X^s, a_i^s i\in A^s, y_i^s \in Y^s\} $

This represents the set of seen data. It refers to the set of seen data $x_i^s$. The variables $X$, $A$, and $Y$ belong to sets representing 'input data,' 'semantic representation,' and 'label (class)' data, respectively.

> $ U = {\{(x_j^u, a_j^u, y_j^u)_{j=1}^{N_u}} \mid x_j^s \in X^s, a_j^s j\in A^s, y_j^s \in Y^s\} $

Similarly, this represents a data set, but the data belonging to $U$ is the set of 'unseen data.'

In thi part,
$$
{x_i}^s,{x_j}^u \in R^D
$$
Definition like this one appears. It implies that for both seen data and unseen data, the input data $x$ is assumed to be in a $D$-dimensional vector space. What this means is that unseen data also resides in the same dimensional vector space as seen data. Suppose there is a pre-trained model that has already passed image data through a deep learning neural network to reduce its dimensionality in a way that well represents the characteristics of a 'horse.' To predict or generate images of a horse that the model hasn't seen before (for example, a zebra), the unseen data input must also be of the same dimensionality.

> ${x_i}^s,{x_j}^u \in R^D$ indicate the D-dimensional images (visual features) in the feature space $mathcal{X}$ that can be obtained using a pre-trained deep learning model such as ResNet [28], VGG-19 [29], GoogLeNet [30].

If you have trained a model using ResNet, GoogleNet, or similar, to capture the characteristics of the animal "horse," this model will map out a feature space ($\mathcal{x}$) that well represents the characteristics of a "horse." That is, the seen and unseen data $x$, both of which are $D$-dimensional data, are assumed to reside in the feature space ($\mathcal{x}$). When any image data is projected into the feature space ($\mathcal{x}$), it will be projected in a way that well represents the features learned by the neural network related to "horse."

Similarly, we assume that the space $\mathcal{A}$, representing semantic representations, and the space $\mathcal{Y}$, representing 'labels,' include the semantic representations and labels of seen and unseen data, denoted as $a$ and $y$. If the feature space $\mathcal{x}$, mentioned earlier, is the space where the representation of specific data is transformed into features through the neural network, then the semantic representation space is a space created through word embedding, for example, while $\mathcal{y}$ is the space where the final classes obtained through the feature and semantic representation information reside.

Now, suppose there is a model that generates images based on text. The neural network model that transforms the features of input text according to the intended purpose projects the data into a space representing the characteristics of the "text" data. The embedding space learned through embedding contains the semantic representation of the text. Finally, the space containing the data of all the images expected to be generated through the text can be seen as $\mathcal{y}$.
$$
f_{GZSL} : \mathcal{X} \rightarrow \mathcal{Y}
$$
Ultimately, GZSL can be seen as a model that predicts labels using semantic information in a data space composed of both seen and unseen data.

In this post, we've briefly covered the overview, and in separate posts, we'll dive deeper into representation learning, Zero-Shot Learning, and Generalized Zero-Shot Learning.

---

# References

[A Review of Generalized Zero-Shot Learning Methods, Pourpanah et al](https://arxiv.org/abs/2011.08641) 

[Language Models are Few-Shot Learners, Brown et al](https://arxiv.org/abs/2005.14165) 

