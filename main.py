import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
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
    text = replace(text, 'w/', 'with')
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
test['text'] = test["text"].apply(lambda x: clean_text(x))

Xv = val.text
Yv = val.sentiment
Xt = train.text
Yt = train.sentiment

encoder = LabelEncoder()
Yv = Yv.apply(encoder.fit_transform)
Yt = Yt.apply(encoder.fit_transform)

vocab_size = 10000
embedding_dim = 64
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(Xv)
word_index = tokenizer.word_index
sequencesXv = tokenizer.texts_to_sequences(Xv)
paddedXv = pad_sequences(sequencesXv, maxlen=max_length, padding=padding_type, truncating=trunc_type)

sequencesXt = tokenizer.texts_to_sequences(Xt)
paddedXt = pad_sequences(sequencesXv, maxlen=max_length, padding=padding_type, truncating=trunc_type)


BUFFER_SIZE = 10000
BATCH_SIZE = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 5
history = model.fit(paddedXv, Yv, epochs=NUM_EPOCHS, validation_data=(Xt, Yt), verbose=1)

results = model.evaluate(paddedXt, Yt, batch_size=BATCH_SIZE)

gc.collect()

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')





