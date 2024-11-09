---
layout: post
title:  "[DS / ML Back to the Basics Series #1] Statistical implication of Parameter Estimation and Standard Error."
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date: 2024-11-01
last_modified_at: 2024-11-01
categories: Statistics DataScience
published: True
---



**🗒️Related Post: [MLE - Maximum Likelihood Estimator](https://ethhong.github.io/statistics/datascience/2024/08/10/Reviewing-MLE-(Maximum-Liklihood-Estimator).html)**

Image: A spoiler for today's topic

<img width="585" alt="Screenshot 2024-11-07 at 2 43 31 PM" src="https://github.com/user-attachments/assets/be4dc335-990a-4d56-b239-89c898b4081d">

In the first post of **DS / ML Back to the Basics** series, I would like to go over what does it mean when we do the 'modeling', and how do we evaluate if the model is good, or bad. I will use 'mean estimator' to explain statistical implications, and cover Simple Linear Regression model in the follow-up post. Maybe 'mean estimation' could be seem as very basic concept, **but I will focus more on the statistical implication of estimation, and standard error.**

---

# What does is mean to do 'modeling?'

When I was an undergraduate student, I learned about a variety of machine learning models that were transforming the field. I was trained to implement these models in Python and apply them to sample datasets. It felt like magic: you write code, input training data, train the model, and it predicts the result.

However, after working as a full-time product manager in the IT industry for three years, I discovered that in real-world business scenarios, simply applying these models rarely works as expected.

Through my experience in ML projects in an AI lab, collaborating with data scientists as a product manager, and studying statistical analysis in depth, I now understand that this happens when **predictive models are used without understanding the implications behind them.** Modeling is not about building a magical pipeline that produces perfect outputs just because the model is the latest and most advanced. It's about ***1) effectively predicting real-world statistical patterns, 2) using a sample dataset that ideally reflects those patterns, and 3) minimizing possible errors.***

---

# Let's think of modeling as 'estimation'

<img width="658" alt="image" src="https://github.com/user-attachments/assets/6ca73492-36de-440d-825f-4cb0536645eb">

We talked a little bit about this in the previous post - [MLE - Maximum Likelihood Estimator](https://ethhong.github.io/statistics/datascience/2024/08/10/Reviewing-MLE-(Maximum-Liklihood-Estimator).html). We identify certain patterns from observed phenomena. While we cannot be certain where these patterns originate or what causes them, if we can estimate the patterns accurately, it may be possible to make valid predictions about unseen data.

## Example of 'mean' prediction.

We learned a lot about sample means and population means in statistics class—even back in high school. Let's think of the sample mean in terms of "estimation." For example, let’s say we want to answer the question, "How heavy are elephants?" This might be a difficult question because individual elephants vary in weight. However, we definitely know that elephants are much heavier than dogs or cats.

The best way to describe "on average, how heavy are elephants?" would be to take the mean weight of all elephants. Since it’s impossible to measure the entire population, we use a sample. Here, we can say the "sample mean" is an estimator for the "population mean"—implying that the "population mean" is the actual (or ideal) real-world pattern describing the average weight of elephants, and we are estimating it through the "sample mean."

Based on what we learned in statistics classes, assuming that weight is normally distributed, we can say:

* Random variable for the weight of elephants: $Y \sim N(\mu, \sigma)$
* Sample variable follows: $\bar{Y}, s_Y$

Here, $\mu$ is the population mean (i.e., the actual average weight of elephants as determined by some real-world pattern that we cannot observe directly), while $\bar{Y}$ is an estimator for what we’re interested in.

So, what we have done here is **estimate the statistic of a real-world pattern using a limited sample dataset.** This is what we do for every "estimation" or modeling exercise. But we still have one question—"Can we just accept this estimator? How precise is it?"

## Evaluating estimation.

What should we examine to assess the precision of an estimation? Let’s think of it this way: look at the two cases below. Both cases have the same mean; however, the data in the second case is much more spread out, meaning it has a higher variance from the mean. 

---



<img width="585" alt="Screenshot 2024-11-07 at 2 43 31 PM" src="https://github.com/user-attachments/assets/be4dc335-990a-4d56-b239-89c898b4081d">

---

This implies that if the standard deviation is high, data points are more likely to be far from our estimated mean. (In a standard normal distribution, approximately 95% of data falls within 2 * std of the mean.) ***Therefore, if the standard deviation of our estimator is low, we can say our prediction is more efficient in terms of predicting unseen future data!***

### The, how is standard deviation of mean estimator like? - True implication of CLT

Let’s go back to statistics class. Now it’s time for the **Central Limit Theorem** to do its job. Assuming $Y$ is normally distributed, the Central Limit Theorem states that


$$
\bar{Y} \sim N\left(\mu, \frac{\sigma^2}{n}\right)
$$



if the sample size $n$ is large enough. Here, we see that the variance (or standard deviation) of our mean estimator is ***1) determined by the population variance, and 2) divided by sample size.*** What are the implications of this?

1. With a larger sample size, our estimator (sample mean) will be closer to the population mean, resulting in more precise estimation.
2. However, if the population standard deviation itself is large—meaning if $\sigma$ is large—there will be limitations to the precision of mean estimation. 

These implications are more apparent if we consider it this way: as our sample size approaches the population size, we are effectively sampling the entire population. Therefore, as the sample size increases, the sample mean approaches the population mean. **This is why we need large amounts of "BIG" data.**

However, this doesn’t mean our estimation will always be precise just because we use a large dataset. What if the "true pattern" of the real world has a huge variance? For example, ***what if what we’re trying to estimate is not elephant weight, but the "income of people living in California?"*** California has a large population with a wide range of incomes, so even if we take the population mean, it may not adequately represent a "pattern." Therefore, for this type of prediction problem, using a sample mean predictor would not work well, even if we take a large sample from California.

### Why does 'dividing segment' work better when estimating statistics?

So, what can we do in these kinds of cases? Product managers or analysts might suggest **"segmenting the data."** The reason this approach works is that by dividing segments, we can reduce the population variance.

If we divide the "population" into different segments—e.g., by age group—it’s more likely that people within the same age group will have a similar range of income. By defining segmented populations, we can reduce the variance of the population statistic. Although the maximum sample size is reduced (since the population is now divided into segments), this segmentation allows us to make better predictions by effectively narrowing down the population variance.

Let’s try this with a data example. There’s an older dataset that contains wage data along with demographic information—the [ISLR Wage dataset](https://rdrr.io/cran/ISLR/man/Wage.html)—which we’ll analyze using R.

```R
install.packages("ISLR", repos = "http://cran.us.r-project.org")

library(ISLR)

# Load and view Wage dataset
data("Wage")
head(Wage)
```

<img width="759" alt="Screenshot 2024-11-08 at 11 10 04 PM" src="https://github.com/user-attachments/assets/a3f8bed9-186d-4a13-b61a-bc4e434a4cdb">

This is how the data looks like. Let's just take data from 2009 (since this is the most recent year from this dataset), and divide age group.

```R
df <- Wage[Wage$year == max(Wage[, 'year']), ]
df$age_group <- ifelse(df$age < 20, "Immature",
                       ifelse(df$age <= 30, "Twenties",
                              ifelse(df$age <= 60, "Middle aged",
                                     "Senior")))
df <- df[, c('age_group', 'wage')]
head(df)
```

Now, let's compare total variance, and variaces for each age groups!

```R
hist(df$wage, main = paste("All age wage - variance:", round(var(df$wage), 2)), 
     xlab = "Wage", ylab = "Frequency")
# Immature
immature_age = df[df$age_group == "Immature", ]$wage
hist(immature_age, main = paste("Under 20 wage - variance:", round(var(immature_age), 2)), 
     xlab = "Wage", ylab = "Frequency")

#Twenties
twenties_age = df[df$age_group == "Twenties", ]$wage
hist(immature_age, main = paste("20~20 wage - variance:", round(var(twenties_age), 2)), 
     xlab = "Wage", ylab = "Frequency")

#Middle aged
middle_age = df[df$age_group == "Middle aged", ]$wage
hist(middle_age, main = paste("30~59 - variance:", round(var(middle_age), 2)), 
     xlab = "Wage", ylab = "Frequency")
```

![image](https://github.com/user-attachments/assets/093d19c1-0441-4003-a3af-aa7894e7d676)

![image](https://github.com/user-attachments/assets/a10789bb-2bf6-4630-a798-2c7b2ce69567)

![image](https://github.com/user-attachments/assets/0021c2bf-30e7-467e-82dc-2b15144f2ed0)

![image](https://github.com/user-attachments/assets/58c42d6e-ed25-4619-97fe-6626e7be730d)

We can see that the variance for the total population was 1811—a relatively large value. However, by splitting age groups, we could reduce the variance significantly for the under-20 and 20s age groups. If we use the entire population for mean estimation, $\sigma_Y$, the population standard deviation of wage, is $\sqrt{\sigma_Y} = 42.5629$. In comparison, the standard deviation of the 20s age group segment is $\sqrt{\sigma_{Y=20}} = 29.32$.

In a normal distribution, 95% of data is located within $\text{mean} \pm 2\sigma$ (the confidence interval). Therefore, if we use the mean as the estimator, the 95% confidence interval in the first case is about 85, while in the second case, it’s much narrower—around 58.

Interestingly, the middle-aged group has a much larger variance than the total population. **This suggests that we selected an inappropriate segment for mean prediction.** This makes sense, as the 30–59 age range is quite broad for generalizing income patterns. To exaggerate, both myself two years from now and Jeff Bezos would fall into this group! We may reduce the variance further by splitting this group more narrowly or by removing outliers when conducting estimation.

## There's still one more problem - $\Sigma$ is unkonw.

I believe many readers have heard of "Standard Error." Now it’s time to introduce what it is and why it’s important. We just discussed why the "Standard Deviation" of the value we’re trying to estimate (in our example, $\bar{Y}$, which is the average weight of elephants in the world) is relevant. According to the Central Limit Theorem, we know how to determine this value: $\bar{Y} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$. However, the population variance, $\sigma$, is an unknown value. Therefore, **we use the sample standard deviation as an estimator for the population standard deviation.**
$$
\sigma_{\bar{Y}} = \frac{\sigma}{\sqrt{n}} \sim s_{\bar{Y}}= \frac{s_{Y}}{\sqrt{n}}
$$
Therefore, **"Standard Error"** is an estimator of the "Standard Deviation of the value we are trying to estimate."

Remember:

1. If we have a lower standard error, it means our prediction is likely more accurate.
2. To achieve a smaller standard error, we need a larger ***N*** (sample size), but
3. There are limitations if the **"Standard Deviation (or variance) of the population itself"** is large.

Also, since \( n \) in the denominator of the standard error formula is square-rooted, as \( n \) increases, the impact of reducing standard error gradually diminishes.

---

## Coming up next: let's look into 'Simple Linear Regression' model

Now we have a better understanding of the true implications of "estimation" and "standard error." **But I hear someone saying...**

### ***It's just taking average for estimation! It's not a REAL modeling!*** 

<img src="https://github.com/user-attachments/assets/c899c861-268b-4794-b601-3e92dc18f36c" width="400" height="400" />

Now we have a better understanding of the true implications of "estimation" and "standard error." **But I hear someone saying...**
$$
\hat{y} = f(x) = E[X] = \frac{1}{n}\sum_{i=1}^nx_i
$$


It could also be considered a type of model. **BUT, I totally get what you mean.** In the next post, I will use the example of **Linear Regression** to explore how Standard Error works in evaluating regression models and how we can properly utilize regression models for better prediction. (Remember, models are not magic boxes!)
