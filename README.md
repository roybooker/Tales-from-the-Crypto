# Tales from the Crypto

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
