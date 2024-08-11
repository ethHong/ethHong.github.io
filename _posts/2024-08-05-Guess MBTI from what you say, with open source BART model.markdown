---
layout: post
title:  "Build and deploy an MBTI Predictor Using Streamlit and Huggingface Spaces: Utilizing open source BART model"
tags:
  - Blog
  - MathJax
  - Jekyll
  - LaTeX
use_math: true
date: 2024-08-05
last_modified_at: 2024-08-05
categories: BuildForFun LLM
published: true
---

* 🔗[Link to Repository](https://github.com/ethHong/mbti_translator_demo)
* 🔗[Link to Huggingface Spaces](Sukhyun/MBTI_translator)

* Build a MBTI classifier utilizing open source BART text classification model. 
* Use Streamlit and Huggingface Spaces to easily build demo and deploy.

# MBTI Predictor

* I had built this simple app with Python and Streamlit library (Just for fun)
*  It's running on a 🔗[Huggingface Spaces](Sukhyun/MBTI_translator), and if the connection is lost (or kernal is dead), please try to enter the link directly and restart the kernal!
* **How to use?** : Just simply put your sentence, and push 'generate'. It may infer your MBTI type based on the sentence, and scores for each of 4 dimensions. 
  * For instance, when you put ***I stayed home all day***, it gives you 'ISFP'. 
  * I tried with a sentence ***We don't have time for wining. Let's just focus on our work without wasting time.*** and this time it gives you a 'ESTJ'

<iframe
	src="https://sukhyun-mbti-translator.hf.space"
	frameborder="0"
	width="850"
	height="800"
></iframe>

# That sounds like... ~ MBTI type!

For recent several years, MBTI test had been highly trending in Korea (I'm not sure it was also trending in another regions...). MBTI (Myers-Briggs Type Indicator) is one of the methodologies used to indicate one's personality through a questionnaire, which classifies one's personality into 16 different types based on the responses. Even there are some criticism against classifying personality in countable number of categories, it had been kind of cultural trend in young generation.

We sometimes here, people say 'That sound like a ***T***', or 'you sound lik an ***F***'. Well, it's not good to have prejudices or stereotypes, and most of the people are enjoying these for fun, rather than taking it seriously. 

<img width="369" alt="Screenshot 2024-08-11 at 6 26 53 PM" src="https://github.com/user-attachments/assets/4e655c60-8d4b-46e0-9e53-915d4c98ec7a">

However, when I heard about the MBTI, **I recognized that this is a vector with dimension of 1 *4**. There are 4 aspects of personalitiest:

* Extrovert / Introvert
* Sense / Intuition
* Thinking / Feeling
* Judging / Perceiving

For each of the dimensions, everyone may have continuous and numerical score. **Here, I I got an idea to build a prediction module, which matches a sentence, or behavior one make into the most probable MBTI Type.** 

# Approach

## Utilizing text classifers

How does MBTI work? Let's think of MBTI as a 1*4 vector.

$MBTI = [EI, SN, TF, PJ]$

The first dimension $EI$ represents a score, or some numerical value which represent 'how likely the person is an 'Extrovert' rather than 'Introvert'. We can represent rest of the dimension in the same way. 

Then, how can we convert a **text** into a vector which represent MBTI? If there are some keywords that represent each of the MBTI aspects, and see which keywords are semantically clost to the given sentece, we may be able to decide the MBTI class fo the given sentence. 

For instance, we may define **tems like ["extrovert", "expression", "outside", "together"]** to represent 'Extrovert', while **["introvert", "indirect", "Concerns", "alone"] to represent 'Introvert'.**

### Text classification model - BART

I utilized [text inference model](https://huggingface.co/facebook/bart-large-mnli) based on [BART](https://huggingface.co/docs/transformers/model_doc/bart) by Facebook (Currently, Meta). According to the [overview](https://huggingface.co/docs/transformers/model_doc/bart), BART is a sequence-22sequence based pre trained model with bidirectional encoder, which is basically similar to BERT model. It is not one of the most recent models, since it's released in 2019. In current days, LLMs like ChatGPT may outperform in tasks like this. However, I believe using LLM is not always the best just because they are the most recent ones. When I tried on my local device, BART pretrained model is strong, but small enough to run on local devices - even on CPU. Since, what I need for this task is 'getting a probability of certain term closely related to the given text.'.

 <img width="529" alt="Screenshot 2024-08-11 at 10 33 58 PM" src="https://github.com/user-attachments/assets/65a62f14-4e78-461e-84b8-49e01c98780a">

The above is an example provided by Huggingface BART [text inference model](https://huggingface.co/facebook/bart-large-mnli) page. The given model is an zero-shot text classification model, **which is given with 1) A text, 2) and several classes (or words), classifying the given text based on the probability of the text belongs to each of the class labels.** 

## Let't build the translator!

Now, let's try to utilize this model to build a MBTI classifier. First, I tried to define sets of keywords that represent each of 8 MBTI elements. I had created a json file named ~mbti_map.json~. You may customize and test on different sets of keywords. 

~~~json
#mbti_map.json
{
  "E_I": {
    "E": ["extrovert", "expression", "outside", "together"],
    "I": ["introvert", "indirect", "Concerns", "alone"]
  },
  "N_S": {
    "N": ["intuition", "ideal", "theoretical", "prediction"],
    "S": ["sensing", "realistic", "useful", "implimentation"]
  },
  "T_F": {
    "T": ["thinking", "logical", "factful", "objective"],
    "F": ["feeling", "relationship", "value", "sympathy"]
  },
  "P_J": {
    "P": ["perceiving", "elasticity", "autonomy", "indiscriminate"],
    "J": ["judging", "planning", "objective", "systematic"]
  }
}
~~~

Next, in a ~BART_utils.py~ file, I defined all the functions I need. First, we need to set up and load pretrained BART model. I suggest following the guide of [Huggingface](https://huggingface.co/facebook/bart-large-mnli) regarding how to set up and load model. 

~~~python
#BART_utils.py

import numpy as np
from load_data import *
import matplotlib.pyplot as plt
import streamlit as st
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
nli_model = (
    AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    ).cuda()
    if torch.cuda.is_available()
    else AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
)
~~~

Next, define a function which takes a squence (sentence) and label, and **return probability that the sequence belongs to the label.**

~~~ python
#BART_utils.py
def get_prob(sequence, label):
    premise = sequence
    hypothesis = f"This example is {label}."

    # run through model pre-trained on MNLI
    x = tokenizer.encode(
        premise, hypothesis, return_tensors="pt", truncation_strategy="only_first"
    )
    logits = nli_model(x.to(device))[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:, 1]
    return prob_label_is_true[0].item()
~~~

Utilizing this module, conduct series of tasks by deifing functions:

1. `judge_mbti(sequence, labels)` : When taking a **sequence** and **a set of labels**, return probability score of each labels for that sequence. 
2. `conpute_score(text, type)` : When taking a **sequence**, and MBTI dimension you would like to decide (e.g. "E_I", or "N_S" etc...)
   1. Get sum of probability scores for each of the types, using 'judge_mbti'. 
3. `mbti_translator(text)` : Give the final choice, and show scores.

~~~python
#BART_utils.py
def judge_mbti(sequence, labels):
    out = []
    for l in labels:
        temp = get_prob(sequence, l)
        out.append((l, temp))
    out = sorted(out, key=lambda x: x[1], reverse=True)
    return out

def compute_score(text, type):
    x, y = type.split("_")
    x_score = np.sum([i[1] for i in judge_mbti(text, keywords_en[type][x])])
    y_score = np.sum([i[1] for i in judge_mbti(text, keywords_en[type][y])])

    if x_score > y_score:
        choice = x
        score = x_score
    else:
        choice = y
        score = y_score

    x_score_scaled = (x_score / (x_score + y_score)) * 100
    y_score_scaled = (y_score / (x_score + y_score)) * 100

    stat = {x: x_score_scaled, y: y_score_scaled}

    return choice, stat

def mbti_translator(text):
    E_I = compute_score(text, "E_I")
    N_S = compute_score(text, "N_S")
    T_F = compute_score(text, "T_F")
    P_J = compute_score(text, "P_J")

    return (E_I[0] + N_S[0] + T_F[0] + P_J[0]), (E_I[1], N_S[1], T_F[1], P_J[1])

~~~

Adding to this, we can also **plot**, like this to show how closely the sentence is classified as a chosen lable:

<img width="1040" alt="Screenshot 2024-08-11 at 10 55 36 PM" src="https://github.com/user-attachments/assets/a1c0e7fe-c554-488e-b995-de23a673cae8">

~~~python
#BART_utils.py
def plot_mbti(result):
    fig, ax = plt.subplots(figsize=(10, 5))

    start = 0
    x, y = result.values()
    x_type, y_type = result.keys()

    ax.broken_barh([(start, x), (x, x + y)], [10, 9], facecolors=("#FFC5BF", "#D4F0F0"))
    ax.set_ylim(5, 15)
    ax.set_xlim(0, 100)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_yticks([15, 25])
    ax.set_xticks([0, 25, 50, 75, 100])

    ax.text(x - 6, 14.5, x_type + " :" + str(int(x)) + "%", fontsize=15)
    ax.text((x + y) - 6, 14.5, y_type + " :" + str(int(y)) + "%", fontsize=15)

    st.pyplot(fig)
~~~

## Building interface with Streamlit

Working on this project, I was looking for the most simple way to build a demo with UI. Long ago, I have once built an demo UI with Flask and HTML for [🔗another project](https://github.com/ethHong/LDA_Job2Course_Recommender). It was simple, but there were many limitations. Also, since I had almost no background in Web programming, learning all the frameworks required to build simple demo for analytical project was way too time consuming. (But still, I wish to learn web development starting with React some day!)

I found that the [Streamlit](https://streamlit.io) library is one of the greate ways to create a demo, especially if you are working on some analytical projects, or simple apps utilizing AI / ML models. Syntax is simple and easy (I believe it's simpler than HTML), and thanks to expanding community more various components are being updated. 

I am sharing the code of demo I shared at the beginning of the post. (It's only 38 lines of Python codes to build a UI!)

~~~python
#app.py
import streamlit as st
from BART_utils import (
    get_prob,
    judge_mbti,
    compute_score,
    mbti_translator,
    plot_mbti,
    device,
)

st.title("MBTI translator")
if device == "cpu":
    processor = "🖥️"
else:
    processor = "💽"
st.subheader("Running on {}".format(device + processor))

st.header("💻Infer my MBTI from my langauge (What I speak)")
st.write("🤗Give any sentences: I'll try to guess your MBTI")
st.header("🤔How it works??:")
st.write(
    "Using Zero-Shot NLI model, it computes probability of sentence and MBTI keywords"
)
st.write("More about the model: https://github.com/ethHong/mbti_translator_demo")

user_input = st.text_input("👇👇Put your sentence here", "I stayed home all day")
submit = st.button("Generate")

if submit:
    with st.spinner("AI is analysing result..."):
        output_mbti, output_ratio = mbti_translator(user_input)

    st.success("Success")
    st.subheader("🤔Probable MBTI is...🎉 : " + output_mbti)

    for result in output_ratio:
        plot_mbti(result)

~~~