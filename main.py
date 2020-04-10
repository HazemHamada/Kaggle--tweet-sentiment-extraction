import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Bidirectional, Dropout, Flatten
from sklearn import linear_model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.utils as ku
import matplotlib.pyplot as plt
from keras.utils import plot_model
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from sklearn import metrics
import gc
import string
import re

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)


def clean(reg_exp, text):
    text = re.sub(reg_exp, " ", text)

    # replace multiple spaces with one.
    text = re.sub('\s{2,}', ' ', text)

    return text

def remove_urls(text):
    text = clean(r"http\S+", text)
    text = clean(r"www\S+", text)
    text = clean(r"pic.twitter.com\S+", text)

    return text

def basic_clean(text):
    text=remove_urls(text)
    text = clean(r'[\?\.\!]+(?=[\?\.\!])', text) #replace double punctuation with single
    text = clean(r"[^A-Za-z0-9\.\'!\?,\$]", text) #removes unicode characters
    return text

########################################################################################################################

# solution1 (the analytical way)

"""
def predict_analytically(test_subset, sentiment):

    test_subset = test_subset.apply(lambda x: basic_clean(x))
    sentiment = sentiment.apply(lambda x: basic_clean(x))

    sid = SentimentIntensityAnalyzer()
    word_list = []
    i = 0
    for word in test_subset:

        split_text = word.split()
        score_list = []

        if sentiment[i] == 'positive':
            for w in split_text:
                score = sid.polarity_scores(w)['compound']
                score_list.append(score)
                max_index = np.argmax(score_list)
            word_list.append(split_text[max_index])

        elif sentiment[i] == 'negative':
            for w in split_text:
                score = sid.polarity_scores(w)['compound']
                score_list.append(score)
                min_index = np.argmin(score_list)
            word_list.append(split_text[min_index])

        else:
            word_list.append(word)

        i = i + 1
    return word_list


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.dropna()
train = train.drop_duplicates()

test = train.dropna()
test = train.drop_duplicates()

selected_text_train_pred= predict_analytically(train['text'].astype(str), train['sentiment'].astype(str))

submission = pd.read_csv("sample_submission.csv")

test_subset = test['text'].astype(str)
sentiment = test['sentiment'].astype(str)

submission["selected_text"] = predict_analytically(test_subset, sentiment)

"""

########################################################################################################################

# solution2


"""

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

"""

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

train = train.dropna()
train = train.drop_duplicates()
#plt.hist(train.sentiment)


train['text'] = train["text"].apply(lambda x: basic_clean(x))
train['selected_text'] = train["selected_text"].apply(lambda x: basic_clean(x))
# encoder = LabelEncoder()
# X2v = encoder.fit_transform(X2v)
# X2t = encoder.fit_transform(X2t)
train['sentiment'] = train['sentiment'].apply(lambda x: 2 if x == 'positive' else 1 if x == 'neutral' else 0)



########################################################################################################################

# X1 = train.text
# X2 = train.sentiment
# Y = train.selected_text

train['len1'] = train['text'].apply(lambda x:len(x))
train['len2'] = train['selected_text'].apply(lambda x:len(x))

selected_texts = train['selected_text'].astype(str)
all_train_texts = train['text'].astype(str)


#text_locations=pd.Series([])
#for i, s in enumerate(selected_texts):
#    text_locations = all_train_texts[i].find(s)

text_locations = [all_train_texts[i].find(s) for i, s in enumerate(selected_texts)]
train['text_locations'] = text_locations
len1 = train['len1']
len2 = train['len2']
sentiment = train['sentiment']

"""
def get_new_locations(text, selText,ln):

    result = np.zeros(ln)
    j=0
    for i in selText.split():
        result[j] = (text.find(i)+1)
        j = j+1
    result = pd.Series(result)
    return result


mx=0
for i in train['selected_text']:
    if mx<np.alen(i):
        mx=np.alen(i)

locations = pd.Series([])
for _,row in train.iterrows():
    locations.append(get_new_locations(row['text'], row['selected_text'], mx))
"""

#train, test = train_test_split(train, test_size=0.2, random_state=1)

# to predict 'len2'
Y_train1 = train['len2']
X_train1 = train[['sentiment', 'len1']]
X_test = test[['sentiment', 'len1']]

# to predict 'text_location'
Y_train2 = train['text_location']
X_train2 = train[['sentiment', 'len1']]

reg = lgb.LGBMRegressor()
#reg = linear_model.LinearRegression()
reg.fit(X_train1, Y_train1)

predicted1 = np.round(reg.predict(X_test))
predicted1[predicted1 < 1] = 1


reg2 = lgb.LGBMRegressor()
#reg2 = linear_model.LinearRegression()
reg2.fit(X_train2, Y_train2)

predicted2 = np.round(reg2.predict(X_test))
predicted2[predicted2 < 1] = 1

# now predctions are of the form: index of starting character + length of word
predicted = predicted1 + predicted2

sub = test[['textID', 'text']]
sub['preds'] = predicted

sub['text2'] = sub["text"].apply(lambda x: x.split())
text2 = sub['text2']

textx = sub['text'].tolist()
text_sub = [s[int(predicted2.tolist()[ind]):int(predicted2.tolist()[ind])+int(predicted1.tolist()[ind])] for ind, s in enumerate(textx)]

text2 = [l[-int(predicted.tolist()[ind]):] for ind, l in enumerate(text2)]
sub['text22'] = text2
sub['result'] = sub["text22"].apply(lambda x: " ".join(x))

submission["selected_text"] = sub['result']
submission.to_csv('submission.csv', index=False)
########################################################################################################################

"""
vocab_size = 10000
embedding_dim = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X1v)
word_index = tokenizer.word_index

sequencesX1v = tokenizer.texts_to_sequences(X1v)
X1v = pad_sequences(sequencesX1v, maxlen=max_length, padding=padding_type, truncating=trunc_type)

sequencesX1t = tokenizer.texts_to_sequences(X1t)
X1t = pad_sequences(sequencesX1t, maxlen=max_length, padding=padding_type, truncating=trunc_type)

sequencesYv = tokenizer.texts_to_sequences(Yv)
Yv = pad_sequences(sequencesYv, maxlen=max_length, padding=padding_type, truncating=trunc_type)

sequencesYt = tokenizer.texts_to_sequences(Yt)
Yt = pad_sequences(sequencesYt, maxlen=max_length, padding=padding_type, truncating=trunc_type)


Xv = np.column_stack((X1v, X2v))
Xt = np.column_stack((X1t, X2t))

gc.collect()

total_words = len(tokenizer.word_index) + 1
max_length = max_length+1
"""

########################################################################################################################

"""
BATCH_SIZE = 64


model1 = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32, return_sequences=True)),
])


model2 = tf.keras.Sequential()
model2.add(tf.keras.layer.Embedding(total_words, 100, input_length=max_length))
model2.add(Bidirectional(LSTM(150, return_sequences=True)))
model2.add(Dropout(0.2))
model2.add(Bidirectional(LSTM(100, return_sequences=True)))
model2.add(Dense(total_words/2, activation='relu'))
model2.add(Dense(total_words, activation='softmax'))




def seq2seq_model_builder(HIDDEN_DIM=300):

    encoder_inputs = Input(shape=((max_length, 1)), dtype='int32', )
    encoder_embedding = Embedding(total_words, 100, input_length=max_length)(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = Input(shape=((max_length, 1)), dtype='int32', )
    decoder_embedding = Embedding(total_words, 100, input_length=max_length)(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model


model1.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3 = seq2seq_model_builder()
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


NUM_EPOCHS = 2
history1 = model1.fit(np.array(Xv), np.array(Yv), epochs=NUM_EPOCHS, validation_data=(np.array(Xt), np.array(Yt)), verbose=1)
results1 = model1.evaluate(np.array(Xt), np.array(Yt), batch_size=BATCH_SIZE)


history2 = model2.fit(np.array(Xv), np.array(Yv), epochs=NUM_EPOCHS, validation_data=(np.array(Xt), np.array(Yt)), verbose=1)
results2 = model2.evaluate(np.array(Xt), np.array(Yt), batch_size=BATCH_SIZE)


history3 = model3.fit(np.array(Xv), np.array(Yv), epochs=NUM_EPOCHS, validation_data=(np.array(Xt), np.array(Yt)), verbose=1)
results3 = model3.evaluate(np.array(Xt), np.array(Yt), batch_size=BATCH_SIZE)


gc.collect()

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history1, 'accuracy')

plot_graphs(history1, 'loss')







def seq2seq_model_builder(HIDDEN_DIM=300):

    encoder_inputs = Input(shape=(max_length,), dtype='int32', )
    encoder_embedding = Embedding(total_words, 100, input_length=max_length)(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = Input(shape=(max_length,), dtype='int32', )
    decoder_embedding = Embedding(total_words, 100, input_length=max_length)(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model


max_length=100

model3 = seq2seq_model_builder()
model3.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history3 = model3.fit(np.array(Xv), np.array(Yv), epochs=NUM_EPOCHS, validation_data=(np.array(Xt), np.array(Yt)), verbose=1)


history3 = model3.fit([X1v, X2v], Yv, epochs=NUM_EPOCHS, validation_split=0.2, verbose=1)




model4 = Sequential()
model4.add(LSTM(256, input_shape=(Xv.shape[0], X1v.shape[1]), return_sequences=True))
model4.add(Dropout(0.2))
model4.add(LSTM(256, return_sequences=True))
model4.add(Dropout(0.2))
model4.add(LSTM(128))
model4.add(Dropout(0.2))
model4.add(Dense(Yv.shape[1], activation='softmax'))

model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history4 = model3.fit([X1v, X2v], Yv, epochs=NUM_EPOCHS, validation_split=0.2, verbose=1)

"""




