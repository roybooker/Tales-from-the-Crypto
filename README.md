# Unit-11---Risky-Business

### Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked that you help them predict credit risk with machine learning techniques.

### I have used machine learning models to predict credit risk using data you'd typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem, so I have employed different techniques for training and evaluating models with imbalanced classes. I have applied imbalanced-learn and Scikit-learn libraries to build and evaluate models using the Resampling ##

--------

## Technologies

This notebook leverages Python 3.8 with the following packages in a Jupyter Notebook:
```python
import os
import pandas as pd
from dotenv import load_dotenv
import nltk as nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from newsapi import NewsApiClient

%matplotlib inline

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re
from collections import Counter
from nltk import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib as mpl
```

## Installation Guide

Before running the notebook, please install the dependencies:

```python
pip install pandas
pip install sklearn
pip install matplotlib
pip install imblearn

```

## Observations

### Sentiment Analysis ###

Q: Which coin had the highest mean positive score?
    
A: Bitcoin has the highest Mean Positive score of 0.057 which is slightly higher than the mean positive score of Ethereum 0.052

Q: Which coin had the highest compound score?

A: Ethereum has the highest Maximum Compound score of 0.586. At the same time, Bitcoin has the highest Mean Compound score of 0.527

Q. Which coin had the highest positive score?

A: Bitcoin has the highest Maximum Positive score of 0.187.


## Contributors##

By: Roy Booker

---

## License ##

MIT
