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


def replace(text, old, new):
    new_strings = []
    for string in text:
        new_string = string.replace(old, new)
        new_strings.append(new_string)
    return new_strings



def clean_text(text):
    spell = SpellChecker()
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    Stopwords = set(stopwords.words('english'))
    text = text.lower()  # lowercase text
    text = spell.split_words(text)
    # text = text.replace('\t', ' ').replace('\n', ' ')
    text = replace(text, '\t', ' ')
    text = replace(text, '\n', ' ')
    text = [spell.correction(word) for word in text]  # correct the spelling mistakes in the text
    for punctuation in string.punctuation:
        # text = text.replace(punctuation, ' ')
        text = replace(text, punctuation, ' ')  # replacing newlines and punctuations with space
    # text = replace_by_space_re.sub(' ', text)  # replace replace_by_space_re symbols by space in text
    # text = bad_symbols_re.sub('', text)  # delete symbols which are in bad_symbols_re from text
    text = [replace_by_space_re.sub(' ', word) for word in text]
    text = [bad_symbols_re.sub('', word) for word in text]
    # text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = [re.sub(r"\s+[a-zA-Z]\s+", ' ', word) for word in text]
    text = ' '.join(word for word in text if word not in Stopwords)  # delete stopwors from text
    return text


train = train.dropna()
train = train.drop_duplicates()
plt.hist(train.sentiment)

train, val = train_test_split(train, test_size=0.3, random_state=1)
val, test = train_test_split(val, test_size=0.4, random_state=1)

val['text'] = val["text"].apply(lambda x: clean_text(x))









