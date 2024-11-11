#################################
##### 2) Sentiment Analysis #####
#################################

# input: timelines_filtered.json
# output: timelines_sentiments.json

###### 1. prepare session #####
# activate virtual environment "masterenv"
import os
activate_this = os.path.join("/home/mueller/MA_researcher_wellbeing/masterenv", "bin", "activate_this.py")
exec(open(activate_this).read(), {'__file__': activate_this})

# import necessary packages
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import numpy as np
import pandas as pd
import urllib.request
import csv
import re
import json
import tqdm
import torch

# load dataset in chunks to avoid memory overload
# timelines_filtered = twitter timelines after preprocessing/filtering (see figure 2)
timelines_filtered = pd.DataFrame()
for chunk in tqdm.tqdm(pd.read_json("/home/mueller/MA_researcher_wellbeing/data/timelines_filtered.json", lines=True, chunksize=200000)):
    timelines_filtered = pd.concat([timelines_filtered, chunk])



##### 2. set everything to use pre-trained roBERTa based model for sentiment analysis #####
# define function for preprocessing of tweets
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

# set model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)



##### 3. Sentiment Analysis #####
l = len(timelines_filtered)
neg_sents, neutr_sents, pos_sents = [], [], []

# preprocess tweets and then iterate through them for sentiment analysis
preprocessed_texts = [preprocess(timelines_filtered.iloc[i, 2]) for i in range(l)]

for i, text in enumerate(preprocessed_texts):
    try:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        neg, neutr, pos = scores
        neg_sents.append(neg)
        neutr_sents.append(neutr)
        pos_sents.append(pos)
    except Exception as e:
        print(f"Error processing tweet {i}: {str(e)}")
        neg_sents.append(None)
        neutr_sents.append(None)
        pos_sents.append(None)
    print(i)

# add list of sentiments to df
timelines_filtered["neg_sent"] = neg_sents  
timelines_filtered["neutr_sent"] = neutr_sents  
timelines_filtered["pos_sent"] = pos_sents 


### save df as JSON!
timelines_filtered.to_json("/home/mueller/MA_researcher_wellbeing/data/timelines_sentiments.json", orient = "records", lines = True)