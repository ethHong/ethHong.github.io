---
layout: post
title:  "World Hapiness Data - Modeling MLE with Python"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date: 2024-08-29
last_modified_at: 2024-08-29
categories: Statistics DataScience
published: True
---

**🗒️Past post on MLE: [Link](https://ethhong.github.io/statistics/datascience/2024/08/10/Reviewing-MLE-(Maximum-Liklihood-Estimator).html)**

**🔗[Github Repo](https://github.com/ethHong/MLE_regression_world_happiness): Refer to the [main.ipynb](https://github.com/ethHong/MLE_regression_world_happiness/blob/main/main.ipynb) notebook.** 

# Now, Let's try to model MLE using Python

From the last post, I answered to the following questions:

1. What is MLE, and what is 'maximizing likelihood' imply?
2. How could MLE be related to Bayesian approach?
3. How is MLE related to Linear regression?

---

I figured out that when we tried to maximize likelihood of the seen data coming from certain model, and likelihood of the model itself, 
$$
P(Data | Model) * P(Model)
$$
we finally get linear regression with regularization term - which Find estimator (model parameters) with least square error, while Preventing overfitting (not resulting in too unlikely model, which only fits for too specific cases!)

<img width="734" alt="image" src="https://github.com/user-attachments/assets/51f86905-d4a5-4451-abc7-5bc3becbb080">

Therefore, if we model MLE with Python and compare the result it Scikit-learn linear regression model, the result is expected to be highly relevant. (Maybe they might not be exactly same, since there would be minor differences in parameters as learning rate.)

---

# Dataset - World happiness data.

🔗[World Hapiness Report Data](https://worldhappiness.report/data/)

## First look into data

First, let's try to look into the dataset. Load reqruired libraries, and inspect what kind of information are included.

~~~python
#Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing data
df = pd.read_csv("data/world_happiness.csv")

#Column lists
print (df.columns)

#We need to refine columns names which contain strings
df.columns = ["_".join(i.split(" ")).lower() for i in df.columns]
print("Percentage of NANs:\n\n", round(df.isna().sum()/df.shape[0] * 100, 2))

#Drop data with NAN 
df = df.dropna()
df.head()

#Inspect with Pairplot 
sns.pairplot(df)
~~~

![image](https://github.com/user-attachments/assets/070aafb1-a469-429b-b70c-f13381596421)

### Removing outliers

We can figure out that the data includes following features: 

['Country name', 'year', 'Life Ladder', 'Log GDP per capita',       'Social support', 'Healthy life expectancy at birth',       'Freedom to make life choices', 'Generosity',       'Perceptions of corruption', 'Positive affect', 'Negative affect']

These data seems to be demographical data, and some quantified information related to quality of life. From the scatterplot, we could see that there are some spots which seems to be outliers.

1. For outliers, for each of features we are removing rows which are deviating Q1~Q3 iqr.
2. After that, we are inspecting correlations between each variables

~~~python
# Define functions to easily remove outliers.
def iqr_threashold(df, colname): 
    q1 = df[colname].quantile(0.25)
    q3 = df[colname].quantile(0.75)
    iqr = q3-q1
    
    min = q1-1.5*iqr
    max = q3+1.5*iqr
    return (min, max)

def check_outliers(df, colname):
    iqr_min, iqr_max = iqr_threashold(df, colname)[0], iqr_threashold(df, colname)
    return df[(df[colname] < iqr_min) | (df[colname] > iqr_max)].shape[0]

def filter_for_column(df, colname):
    iqr_min, iqr_max = iqr_threashold(df, colname)[0], iqr_threashold(df, colname)
    return df[(df[colname] >= iqr_min) & (df[colname] <= iqr_max)]
  
outlier_columns = [
    i for i in df.columns 
    if np.issubdtype(df[i].dtype, np.number)  # Check if the column is numeric
    and check_outliers(df, i) >= 0  # Check for outliers
    and i not in ["country_name", "year"]  # Exclude specific columns
]

temp = df
for i in outlier_columns:
    temp = filter_for_column(temp, i)

df = temp
~~~



### Look into correlation between features

After removing outliers, lets briefly look into correlations between features.

~~~python
correlations = df.corr().drop(index = "year", columns= "year")
# Get some highlights
corr = {}
for feature in correlations.columns:
    corr[feature] = dict(correlations[feature][((correlations[feature] >= 0.7)| (correlations[feature] <= -0.7)) & (correlations[feature] != 1)])

corr
~~~

And, this is what we get. I printed features which show high positive, or negative correlation. 

~~~json
{'life_ladder': {'log_gdp_per_capita': 0.7459340901989036,
  'healthy_life_expectancy_at_birth': 0.7141190001571981},
 'log_gdp_per_capita': {'life_ladder': 0.7459340901989036,
  'healthy_life_expectancy_at_birth': 0.8354173024798083},
 'social_support': {},
 'healthy_life_expectancy_at_birth': {'life_ladder': 0.7141190001571981,
  'log_gdp_per_capita': 0.8354173024798083},
 'freedom_to_make_life_choices': {},
 'generosity': {},
 'perceptions_of_corruption': {},
 'positive_affect': {},
 'negative_affect': {}}
~~~

This seems quite intuitive. 'Life ladder', which seems to be highly representing quality (or happiness) of life show high correlation with features relevant to GDP, and life expectancy. However, we should keep in mind that just having correlation does not necessarily imply 'causality'. Even so, features with high correlation are more likely to be important features when modeling.

---

# Try MLE modeling with the data.

Let's pick one of the features - **healthy_life_expectancy_at_birth** as target feature, and make a predictive model. 

🗒️Note - I will mention later again, but one more question here: if we successfully model a MLE targeting healthy_life_expectancy_at_birth, does it mean we can really 'predict' life expectancy based on given features? Can we say those features actually 'caused' certain amount of life expectancy?

---

~~~python
#Split dataset
y = df['healthy_life_expectancy_at_birth'].values

X = df[['log_gdp_per_capita', 'social_support',
        'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption',
        'positive_affect', 'negative_affect']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
~~~

We will define 'regularized_log_likelihood' function which formulates log likelihood of parameter $\theta$, given the dataset. (Past post on MLE:  [Link](https://ethhong.github.io/statistics/datascience/2024/08/10/Reviewing-MLE-(Maximum-Liklihood-Estimator).html))

~~~python
from scipy.optimize import minimize

def regularized_log_likelihood(params, X, y, lambda_reg):
    theta = params[:-1]
    sigma2 = params[-1]
    if sigma2 <= 0:
    	return np.inf
    
    #Prediction
    y_pred = np.dot(X, theta)
    #You can also to X @ beta
    
    # Log-likelihood
    n = len(y)
    log_likelohood = -n/2 * np.log(2 * np.pi * sigma2) - 1/(2 * sigma2) * np.sum((y - y_pred)**2)
    
    # Reg
    reg_term = (lambda_reg / 2) * np.sum(theta[1:]**2)
    
    return -(log_likelohood - reg_term)
~~~

## [UPDATED] About $\sigma$ as a parameter

At the vert first, I found the MLE estimation is not quite working well, so tried to fixing $\sigma = 1$, making assumption that the distribution of resiaul is Gaussian Normal (Which actually, does not make sense). It seemed like working well in the scatter plot, but after studying a little deeper, I found that it is never a correct way. $\sigma$ should not be a constant, but also should be a parameter to optimize!!
$$
\hat{\sigma_\epsilon}^2 = s^2 = \frac{1}{N-2}\sum_{i=1}^{N}e_i^2 = \frac{SSE}{N-2}
$$
As the formulation above shows, $\sigma$ is the estimator of residuals between the true regression line and the population data. Since we don't have information about this $\hat{\sigma_{\epsilon}}^2$ we are using the sample error's standard deviation (regression standard error) as an estimator. Optimized $s^2$ will tell you about 'how is the variance of your regression, and the actual data distribution?' - in other words, how 'accurate' your data is. I will try to post about the estimation and regression in the future, to talk about $\sigma$ in detail.

~~~python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.column_stack((np.ones(X_train_scaled.shape[0]), X_train_scaled))
X_test_scaled = np.column_stack((np.ones(X_test_scaled.shape[0]), X_test_scaled))

lambda_reg = 0.1
initial_guess = np.ones(X_train_scaled.shape[1]+1)

result = minimize(regularized_log_likelihood, initial_guess, args=(X_train_scaled, y_train, lambda_reg))

theta_hat = result.x[:-1]
sigma2_hat = result.x[-1]

print("Estimated coefficients:", theta_hat)
print("Estimated variance (sigma^2):", sigma2_hat)
~~~



Now, we have result! We got estimated parameters!

~~~
>>> Estimated coefficients: [63.10162948  5.07081177  0.54476513  0.34363902  0.25787583 -0.07674608
 -0.07708015  0.8429842 ]
 
>>> Estimated variance (sigma^2): 11.277493411971397
~~~

Let's evaluate the results with test data. 

~~~python
## Evaluation
y_test_pred = np.dot(X_test_scaled, theta_hat)
from sklearn.metrics import mean_squared_error

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print ("Test set RMSE: ",rmse_test)
~~~

~~~
>>> Test set RMSE:  3.510943998386364
~~~

### Some interpretation

Here, we have estimated $\sigma^2$ (11.277493411971397) and RMSE (3.510943998386364). RMSE represents how much is the average error, from your prediction and the actual data. It could be meaningful because it tells you how is the 'average' error, but it does not give you the 'confidence' about your model itself. If RMSE is 3.5, is it high or low? Can you ensure that your next prediction for certain X value would be around the line, with similar error?

$\sigma^2$ is what actually tells you about this. $\sigma^2$ implies that, 95% of the data points will be in the range of your prediction $\pm  2* \sigma$. Therefore, lower $\sigma^2$ means your prediction could be more accurate.

## Scatter plot!

~~~python
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted vs Actual')
plt.xlabel('Actual values (y_test)')
plt.ylabel('Predicted values (y_test_pred)')

coeffs = np.polyfit(y_test, y_test_pred, 1)  # Polyfit and polyval: yse leaast square method to fit polynomial data to linear
y_test_pred_line = np.polyval(coeffs, y_test)
plt.plot(y_test, y_test_pred_line, color='green', label='Regression line')
~~~

![image](https://github.com/user-attachments/assets/48c31959-d1ed-42b8-80f8-bf024f8bb5f8)

---

# **Try Scikit-learn Linear Regression and compare result!**

Now, let's try to fit linear regression model - the one we see many times in ML tutorials.

Detailed optimization parameters may have some difference, but since now we know that Linear Regression (which is minimizing square error) with regularization term is identical to likelihood maximization approach, they should show similar outcome. Codes for this part is much simpler, because we are using Scikit-learn library. 

~~~python
#Simply fit Linear Regression to Scikit-learn model
lr = LinearRegression().fit(X_train_scaled, y_train)
w = lr.coef_
b= lr.intercept_

y_test_pred_LL = lr.predict(X_test_scaled)

mse_test_LL = mean_squared_error(y_test, y_test_pred_LL)
rmse_test_LL = np.sqrt(mse_test_LL)

print("Estimated coefficients:", w)
print ("Test set RMSE: ",rmse_test_LL)
~~~

~~~
>>> Estimated coefficients: [ 0.          5.07144864  0.54438915  0.34361327  0.25797011 -0.07666679
 -0.07703181  0.84301896]
 
>>> Test set RMSE:  3.511075850682932
~~~

We can see the test set RMSE  is very similar with the one from MLE model. Also, Even though there are some differences, the weights given my MLE and linear regression are quite close, except for the first feature.

~~~
MLE:
Estimated coefficients: [63.10162948  5.07081177  0.54476513  0.34363902  0.25787583 -0.07674608
 -0.07708015  0.8429842 ]
 
Linear Regression:
Estimated coefficients: [ 0.          5.07144864  0.54438915  0.34361327  0.25797011 -0.07666679
 -0.07703181  0.84301896]
~~~

I believe if we added and tuned regularization in Linear Regression model, the result might have been closer. 

---

# Conclusion

With formula, we figured out that MLE is same as Linear Regression with L2 regularization. And this time we also investigated it through a real-world data. While working on this, I had one more lesson from the experience that **assuming $\sigma = 1$ worked much better for real-world data.** 

However, there still exists some questions to me - Why? Our real world data is not perfectly 'normal'. Therefore I just intuitively thought, there could be more precise, and better distribution that could explain our real-world data. Therefore, rather than just assuming gaussian normal, and standardizing dataset, I thought 'estimating the optimal $\sigma$ value' would give a better performance. I guess this is closely related to simplicity of the model, and Central Limit Theorem - so in case of real world data with huge sample size, Normal Distribution approximates the data quite well. 

**I am planning to do some more experiments and futrher inspection related to this topic some times later!**
