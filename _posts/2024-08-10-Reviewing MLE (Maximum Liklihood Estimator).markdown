---
layout: post
title:  "MLE (Maximum Liklihood Estimator), Bayesian, and Linear Regression - Data Science Back to the Basics"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date: 2024-08-10
last_modified_at: 2024-08-10
categories: Statistics DataScience
published: True
---

# This post review concepts and implications of MLE. 

## Python to model MLE In the next post

In the next post, I will try to use Python to model MLE, and compare results with Linear Regression model in Scikit Learn. 

[**🔗The codes are in this Github Repo](https://github.com/ethHong/MLE_regression_world_happiness),** utilizing [World Hapiness Report Data](https://worldhappiness.report/data/)

This Article and Python practice is made after reviewing [Probability & Statistics for Machine Learning & Data Science](https://www.coursera.org/learn/machine-learning-probability-and-statistics) by DeepLearning.AI in Coursera. It would be great to look into this course if you have time. 

The Python codes and contents of this post is written by the author - Sukhyun Hong after reviewing and understanding the course above. 

---

In undergraduate statistic courses, and major courses as Econometrics, I remember learning MLE (Maximum Likelihood Estimator) and it's relevace with linear regression. However, even I dealt with formulations and solving problems, it was difficult to get the implication of 'maximizing the likelihood', and why it's so important in Machine Learning. 

In this post (and through series of follow up posts) I would like to go through detailed review of key concepts of MLE, Bayesian approaches and Linear regression **not only to just get the formulations, but also look into what they truly imply.** 

In the first part, I would like to review implication, definition, and formulation of MLE with Gaussian Bernoulli.

In the second part, I would like to go through Linear Model with MLE, and go through some Bayesian approach to look into relationship between MLE, Linear Regression.

#  Back to the basics - Maximum likelihood Estimation

## Motivation of MLE / Definition

> Estimating the most 'Likely' interpretation / model / distribution, based on the sample data.

<img width="658" alt="image" src="https://github.com/user-attachments/assets/6ca73492-36de-440d-825f-4cb0536645eb">

Motivation of MLE is answer to **'Which model / distribution / or parameters have the most likely produced the observed data?'** Assume we have some sample data. In most of the cases, what we want to do is identifying the patterns of some behaviour, or group from which the sample are derived in order to predict future behavior. 

This process is basically, maximizing
$$
P(Data|Parameter)
$$
Which is, data comming from specific model, or parameter (e.g : Mean and Variance of normal distritbution, Parameters of linear model...).

## Gaussian example

For instance, you are in a credit card company, and investigating patterns of customers in certain city (let's say it's LA.) Well, it might be also possible to just query all the customers in which `region == "LA"`, but this is not what we want here. We are trying to identify some pattern (distribution) that characterize 'customers in LA'. So, based on  monthly credit card usage data sampled from LA customers, you are trying to estimate **the most likely distribution - mean and standard deviation - of LA customers.**

So, what does 'the most likely distribution' implies? Let's say you have some samples that follows standardized Gaussian normal distribution. <img width="730" alt="image" src="https://github.com/user-attachments/assets/c59cda1a-39c3-4126-931c-f0e0ebed8264">

We have standardized the sample data of LA customers (for simpliciy, let's just say we have 2 samples) : -1 and 1. Among two candidate distribution, $N(50, 1^2)$ and $N(2, 1^2)$ we can intuitively see that it's more likely to have come from the second candidate. This implicate if we are modeling a distribution to describe LA customers' pattern, choosing second distribution is better choice. 

### How can we formulate this 'Likelihood?'

How can we compute the 'likelihood' of the data comming from each distributions? 

1. If we know the PDF (or PMF) of the distribution
2. We can get probability of each data points comming from each candidate distributions
3. And finally, by multipying all probability, we can get the probability of all data comming from the candidate distribution. 

Here, we may notice that we need to assume ***1) Random variables are independent***, and  ***2) For this Gaussian example, we need to assume that data follows normal distribution.***

Let's generalize this Gaussian MLE example. 

We have n samples from Gaussian distribution, which follows unknown parameters (mean, variance) $\mu, \sigma^2$.
$$
X = (X_1, X_2, ...X_n)
$$
Given data sample $X$, likihood of the samples coming from distribution $N(\mu, \sigma^2)$ could be formulated as **multiplication of probability of getting $x_i$ from the distribution**, which is:
$$
L(\mu, \sigma ; x) = \prod\limits_{i=1}^n f_{X_i}(x_i) = \prod\limits_{i=1}^n \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{(x_i-\mu)}{\sigma})^2}
$$
(Using PDF of normal distribution, standardizing into Gaussian Normal.)

This could be written as:
$$
= \frac{1}{\sigma^n(\sqrt{2\pi})^n}e^{-\frac{1}{2}(\frac{\sum_{i=1}^n(x_i-\mu)^2}{\sigma^2})}
$$
Here, if we find $\mu$ and $\sigma$ that maximized this $L(\mu, \sigma ; x)$

To get derivative easility, we can take log-likelihood:
$$
log(L(\mu, \sigma ; x)) = -log(\sigma^n(\sqrt{2\pi})^n)-\frac{1}{2}(\frac{\sum\\_{i=1}^n(x\\_i-\mu)^2}{\sigma^2}) \\\
= -\frac{n}{2}log(2\pi)-nlog(\sigma) - \frac{1}{2}(\frac{\sum\\_{i=1}^n(x\\_i-\mu)^2}{\sigma^2})
$$


We can compute derivative in terms of $\mu$ and $sigma$, like this:

<img width="757" alt="image" src="https://github.com/user-attachments/assets/85da5343-eb03-4b81-b78f-38bf913e8e0e">

In conclusion, $\mu$ that makes
$$
\frac{1}{\sigma^2}(\sum_{i=1}^nx_i - n\mu)  = 0
$$
is 
$$
\frac{\sum_{i=1}^nx_i}{n} = \bar{x}
$$


which means that $\hat{\mu}$, the MLE extimator for population mean is the sample mean $\bar{x}$. 

and $\sigma$ that makes
$$
\frac{n}{\sigma} + (\sum_{i=1}^n(x_i-\mu)^2)\frac{1}{\sigma^3} = 0
$$
is: 
$$
\sigma^2  = \frac{\sum_{i=1}^n(x_i-\mu)^2)}{n}
$$
Replacing $\mu$ with the MLE estimator, we can write that:
$$
\hat{\sigma}  = \sqrt{\frac{\sum_{i=1}^n(x_i-\bar{x})^2)}{n}}
$$
**Here is an implication:**

1. When you don't know population mean and variance, sample mean is the best estimator for the population mean. 
2. Estimator of the sample variance looks very similar to the sample variance, but it takes /n insteats of /(n-1).
3. This assumes that data follows normal distribution, and they are independent.

## Bernoulli Example

Now, we know that the Maximum Likelihood approach is Formulating the likelihood, which is 
$$
P(Data|Parameter / Model / Distribution)
$$
This could be also simply be applied to bernoulli example. 

1. In events that follows Bernoulli Distribution $~Bernoulli(p)$, in which $p$ is probability of success, 
2. When data is given
3. What is the most probable $p$?

For instance, you **tossed a coin 10 times.** If it's fair coin, it should show head : tail ratio close to 50:50. However, assume that you see 9 heads, and 1 tails. In this case, you may be suspicious that the coin is not a fair coin, having $P(H) >0.5$. Then, what is the probability $p$, that maximized $L(p;9H)$?

We can write the Likelihood as:
$$
L(p;9H) = p^9(1-p)^1
$$
Let's generalize this case:
$$
X = (X_1, X_2, ...X_n) \sim^{iid} Bernoulli(p)
$$
where $x$ is 1 of head, and 0 if tail. Like we have done in Gaussian example, Likelihood could be formulated as product of probabilities of $x_i$:
$$
L(p;x) = P_p(X=x) = \prod\limits_{i=1}^npx_i(x_i) = \prod\limits_{i=1}^np^{(x_i)}(1-p)^{(1-x_i)}
$$

$$
= p^{(\sum\limits_{i=1}^nx_i)}(1-p)^{(n - \sum\limits_{i=1}^nx_i)}
$$

Taking log, we get
$$
(\sum\limits_{i=1}^nx_i)log(p) + (n - \sum\limits_{i=1}^nx_i)log(1-p)
$$
and taking a derivative, we have to solve: 
$$
\frac{\sum_{i=1}^nx_i}{p} + \frac{n-\sum_{i=1}^nx_i}{1-p}(-1) = 0
$$
Here, we get $p$ that maximized likelihood:
$$
\hat{p} = \frac{\sum_{i=1}^nx_i}{n} = \bar{x} = \frac{k}{n}
$$
Which is **the number of getting heads.** 

# Linear Regression and MLE

Now, lets try to apply MLE in linear regression case. In this case, given that some data, we are trying to estimate the best fitting model (= model that is most likely to produce given data.) 
$$
P(Data|Model)
$$
Before stating, I would like to pose 2 questions, that you might have heard, but did not have clear answer (at least for me...)

1. Linear regression is about minizing Least Square Errors. How is it related to maximizing Likelihood?
2. Why do we have to assume **independence** of varaibles, and residuals ($\hat{y}-y$) to follow normal distribution? 

**I believe understanding MLE approaches help answering these questions.**

## Maximizing likelihood of generating each points.

Here, what we have to get is likelihood of the data generated from the linear model. Let's say we have a linear model $y=ax+b$, and for each $x_i$, $\hat{y}-y$ is $d_i$

Here, assuming each errors $d_i$ follows Gaussian Normal, we can get likelihood of each data points being sampled from Gaussian Normal distribution. 

<img width="762" alt="image" src="https://github.com/user-attachments/assets/2cb09692-865c-4c83-bf8f-8104d768b584">

I can be written as:
$$
\prod_{i=1}\frac{1}{\sqrt{2\pi}}e^-\frac{1}{2}d_i^2
$$
using PDF of Gaussian normal, as we have done in the Gaussian example.

If we take log likelihood, it becomes, 
$$
e^{-\frac{1}{2}(\sum d_i^2)}
$$
and **maximizing** this term equals to **minimizing** $\sum d_i^2)$

This is identical to getting **Least Square Error**, which is what Linear Regression Do!

# Linear regression, Regularization and Bayesian approach

In linear regression, I believe many of people heard of using 'resularization term' to prevent overfitting. Well, we know that adding regularization term prevent overfitting, but **what does it really imply?** Taking Bayesian approach helps understanding how does adding regularization lead to choosing the most 'likely (or probable)' model.

## Is Model is Maximum Likelihood always good?

We have tried to choose model which maximized likelihood, which is 
$$
P(Data | Model)
$$
However, this may lead to the **overfitting**. What becomes problematic is, the **model** chosen by maximum likelihood, **is less likely to appear.** I think illustration like the one below is used to explain overfitting. Model B may have higher $P(Data | Model)$. But the problem is, it is likely to have models like Model B in our real world case. 

<img width="734" alt="image" src="https://github.com/user-attachments/assets/51f86905-d4a5-4451-abc7-5bc3becbb080">

This implies that, 
$$
P(Data | Model_A) < P(Data | Model_B)
$$
but, 
$$
P(Model_A) > P(Model_B)
$$
Even Model B shows high liklelihood, if the probability of Model B itself is very low, choosing B is not a good option for modeling real-world data. So, we need to consider maximizing :
$$
P(Data | Model) * P(Model)
$$
This is the term we see in **Bayesian Statistic.** (I review Bayesian Statistic and MAP - Maximum A Posterior in another post in detail.)

As we figured out from MLE, $P(Data | Model)$ coule be formulated as : 
$$
\prod\limits_i \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}d_i^2}
$$
Now, we need to get $P(Model)$, which is the probability of the linear model itself. Here, we assume ***parameters are from standard normal distritbution.*** If linear (or polynomial) model is:
$$
y = \theta_0 + \theta_ix_1 + ... \theta_nx_n
$$
We assume 
$$
\theta_i \sim N(0, 1)
$$
Then, by putting $\theta$ into PDF of standard normal, we can write:
$$
\prod\limits_i \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}\theta_i^2}
$$
which is multiplication of probability of $\theta_i$s. So, taking logarithm on $P(Data | Model) * P(Model)$ could be written as:
$$
log(\prod\limits_i \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}\theta_i^2}) + \\
log(\prod\limits_i \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}d_i^2})\\
$$
and maximizing this is same as minimizing 
$$
\sum d_i^2 + \sum \theta_i^2
$$
which is Minimizing **Least Square Error** + **L2 Regularization term.**

To futher extend the formulation, since $d_i$ are errors, we could write
$$
d_i = y_i - \theta_0 - \sum\limits_{j=1}^k \theta_jx_{ij}
$$
So, putting this into $P(Data | Model) * P(Model)$ Likelihood:


$$
logL(\theta, \sigma^2) = -\frac{n}{2}log(2\pi)-nlog(\sigma) \\\
-\frac{1}{2\sigma^2}(\sum_{i=1}^n y_i - \theta_0 - \sum\limits_{j=1}^k \theta_jx_{ij})^2 -\frac{\lambda}{2}(\sum \theta_i^2)
$$

$$

$$



in which $\Lambda$  is a regularizing constant (Weight for regularizaing term.)

## So, this MLE with Bayesian approach lead to Least Square Error + Regularization.

Earlier in Gaussian Example, we had :
$$
log(L(\mu, \sigma ; x)) = -log(\sigma^n(\sqrt{2\pi})^n) -\frac{1}{2}(\frac{\sum_{i=1}^n(x_i-\mu)^2}{\sigma^2}) \\
= -\frac{n}{2}log(2\pi)-nlog(\sigma) - \frac{1}{2}(\frac{\sum_{i=1}^n(x_i-\mu)^2}{\sigma^2})
$$
And the one from Linear Regression example, we had
$$
logL(\theta, \sigma^2) = -\frac{n}{2}log(2\pi)-nlog(\sigma)\\\
- \frac{1}{2\sigma^2}(\sum_{i=1}^n y_i - \theta_0 - \sum\limits_{j=1}^k \theta_jx_{ij})^2 -\frac{\lambda}{2}(\sum \theta_i^2)
$$
Except for having regularization term, we can see they are basically the same. Except for constant terms (in terms of variable $\mu, \sigma, and \theta$) we can see minimizing these formulation is it's basically minimizing 'Errors' between esmimated value, and the actual data. 
$$
\sum_{i=1}^n(x_i-\mu)^2 \text{ for Gaussian,}
$$
and
$$
(\sum_{i=1}^n y_i - \theta_0 - \sum\limits_{j=1}^k \theta_jx_{ij})^2 \text{ for linear model}
$$
