# -*- coding: utf-8 -*-
"""
Persian Sentiment Analysis using Bidirectional LSTM

An end-to-end NLP project for binary sentiment classification of Persian
product reviews using TensorFlow/Keras, Word2Vec, and Hazm.

Author: Mohammad Abbasi
GitHub: https://github.com/shahkooh/persian-sentiment-analysis
"""

from google.colab import drive
drive.mount('/content/drive/')

"""**For Persian language processing in Python**"""

!pip install hazm

# Commented out IPython magic to ensure Python compatibility.
# hazm
from hazm import *


# Matplot
import matplotlib.pyplot as plt
# %matplotlib inline

# DataFrame
import pandas as pd


# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten , MaxPooling1D, LSTM, SpatialDropout1D,GlobalMaxPool1D,Bidirectional
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping , ModelCheckpoint
from keras.models import load_model


# Word2vec
import gensim



# Utility
import re
import numpy as np
import time
import pickle
import itertools

"""**Configs**"""

# DATASET
DATASET_COLUMNS = ['target','text','rate','cat','type','product_id']
DATASET_ENCODING = "UTF-8"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = r"[^آ-ی]"

# WORD2VEC 
W2V_SIZE = 100
W2V_WINDOW = 5
W2V_EPOCH = 32
W2V_MIN_COUNT = 1

# KERAS
SEQUENCE_LENGTH = 100 # for padding
EPOCHS = 100
BATCH_SIZE = 1000

#pre_train model fasttext wikipedia
fasttext_model_pretrain = False

#if use save model set to True
use_save_model = True

"""**Load Dataset**"""

df = pd.read_csv("/content/drive/My Drive/comment.csv",encoding =DATASET_ENCODING, names=DATASET_COLUMNS, header=0)

"""Convert integer target to 'POSITIVE' and 'NEGATIVE'"""

decode_map = {1: "POSITIVE", 0: "NEGATIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
  
df.target = df.target.apply(lambda x: decode_sentiment(x))

#shuffeling
df = df.sample(frac=1).reset_index(drop=True)

df.tail(20)

df

"""**example for matplotlib**"""

# importing the required module 
import matplotlib.pyplot as plt 
  
# x axis values 
x = [1,2,3] 
# corresponding y axis values 
y = [2,4,1] 
  
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('cs_project') 
  
# function to show the plot 
plt.show()

"""**plot distribute of targets**"""

import seaborn as sns
plt.figure(figsize = (6, 8))
sns.countplot(df['target'])
plt.show()

"""**text pre_processing**"""

import io

# read stopwords list
stopwords = []
filepath = "/content/drive/My Drive/stopwords.txt"
with open(filepath, 'r') as f:
    for line in f:
      stopwords.append(line.strip())

"""**stop words and normalizer**"""

normalizer = Normalizer()
stemmer = Stemmer()
def preprocess(text, stem=False):
    text = text.replace('.', ' ')
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    text = text.replace("-","")
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace("«"," ")
    text = text.replace("»"," ")
    text = normalizer.normalize(text)
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text)).strip() # remove all character except farsi character
    
    tokens = []
    # for token in text.split():
    for token in word_tokenize(text):
        if token not in stopwords:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x: preprocess(x))

df.tail(20)

"""**split train test**"""

df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))

word_tokenize("محمد خیلی خوب و مهربان است")

df

"""**convert dataset to document , use for Word2vec embedding**"""

documents = [_text.split() for _text in df_train.text]

print(documents[1])

"""**Create w2v model**"""

w2v_model = gensim.models.Word2Vec(size=W2V_SIZE, 
                                   window=W2V_WINDOW, 
                                   min_count=W2V_MIN_COUNT)

w2v_model.build_vocab(documents)



"""**keras Tokenizer for get dictionary of words and count of words**"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

print(tokenizer.word_counts)

"""**Example of tokenizer and pad_sequences**"""

s = tokenizer.texts_to_sequences([df_train.text[1]])
print(s)
print(tokenizer.sequences_to_texts(s))
print(df_train.text[1])
s_pad = pad_sequences(s,padding='post' ,maxlen=SEQUENCE_LENGTH)
print(s_pad)

"""**set padding for sentence vector to fix vector size**"""

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text),padding='post', maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text),padding='post', maxlen=SEQUENCE_LENGTH)

df_train.head(2)

print(x_train[0])

"""**labels**"""

labels = df_train.target.unique().tolist()
print(labels)

"""**Encoder for lables**"""

encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# print("y_train",y_train.shape)
# print("y_test",y_test.shape)
 

print(y_train[0])
print(x_train[0])

print(y_train[11])
print(y_test[6])

df_train.head(20)

"""**example for dummies**"""

s = pd.Series(list('abca'))
 pd.get_dummies(s)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)

"""**build embedding matrix**"""

embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
unkwon_word_count = 0
uw = []
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
  else:
      uw.append(word)
      unkwon_word_count += 1
print(embedding_matrix.shape)
print(unkwon_word_count)

"""**create embedding layer**"""

# use keras embedding with training
embedding_layer = Embedding(vocab_size, W2V_SIZE,input_length=SEQUENCE_LENGTH,trainable=True)

"""**Build model**"""

model = Sequential()#provides training and inference features on this model.

model.add(embedding_layer)
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(2, activation="softmax"))

# model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15 )
callbacks = [es] # for earlystopping for avoiding overfitting

"""**fit model**"""

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)

# w = model.get_weights()
# w[0]

"""**train evaluation**"""

score = model.evaluate(x_train, y_train, batch_size=100)
print(score)
print()
print("Train ACCURACY:",score[1])
print("Train LOSS:",score[0])

"""**test evaluation**"""

score = model.evaluate(x_test, y_test, batch_size=100)
print(score)
print()
print("Test ACCURACY:",score[1])
print("Test LOSS:",score[0])

"""**confusion matrix plot**"""

from sklearn.metrics import classification_report,confusion_matrix

y_pred = model.predict(x_test,batch_size=1,verbose=2)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

# y_pred = model.predict(x_train,batch_size=1,verbose=2)
# y_train_class = np.argmax(y_train,axis=1)
# y_pred_class = np.argmax(y_pred,axis=1)

# print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('target label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)

    
cnf_matrix = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")
plt.show()

"""**predict one sentence**"""

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    return score

sense_dic = {0:"negative", 1:"positive"}

text = "بوی بدی میده"
res = predict(text)
print(res)
sense_dic[np.argmax(res)]

text = "خیلی خوبه این گوشی "
res = predict(text)
print(res)
sense_dic[np.argmax(res)]

user_test = input("Text : ")
while user_test != "":
  res = predict(user_test)
  print(res)
  print(sense_dic[np.argmax(res)])
  print('------------------')
  user_test = input("Text : ")
