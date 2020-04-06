import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.utils import plot_model
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import gc
import string
import re

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)

train = pd.read_csv("train.csv")


def clean_text(text):
    spell = SpellChecker()
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    Stopwords = set(stopwords.words('english'))
    text = text.str()
    text = text.lower()  # lowercase text
    text = spell.split_words(text)
    text = text.replace('\t', ' ').replace('\n', ' ')  # Single character removal
    text = [spell.correction(word) for word in text]  # correct the spelling mistakes in the text
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')   # replacing newlines and punctuations with space
    text = replace_by_space_re.sub(' ', text)  # replace replace_by_space_re symbols by space in text
    text = bad_symbols_re.sub('', text)  # delete symbols which are in bad_symbols_re from text
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = ' '.join(word for word in text if word not in Stopwords)  # delete stopwors from text
    return text


plt.hist(train.sentiment)









