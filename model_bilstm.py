import matplotlib.pyplot as plt
import itertools
import math
import pandas as pd
import os
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword = stopwords.words('english')
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM, Embedding, Bidirectional, Dropout, Flatten, TimeDistributed
from keras.models import Model, Sequential
from keras.initializers import Constant
import pickle

data = pd.read_csv('100 sentences.csv')

def getUniqueWords(listOfWords):
  uniqueWords = [] 
  for words in listOfWords:
    for word in words:
      if not word in uniqueWords:
        uniqueWords.append(word)

  return uniqueWords

def hasNumbers(inputString):
	return any(char.isdigit() for char in inputString)

transcription_data = []
keywords_data = []
all_data = []
MAX_LEN=0
sentence_column = []
keyword_column = []

# tokenize sentence and label
for transcript, keyword in data[['transcription','keywords']].values:
  if pd.notna(keyword):
    tokenizer = RegexpTokenizer(r'\w+')
    transcript_lower = transcript.lower()
    rem_punct = tokenizer.tokenize(transcript_lower)
    removing_stopwords = ' '.join([word for word in rem_punct if word not in stopword])
    result = ''.join([i for i in removing_stopwords if not i.isdigit()])
    keywords_data.append(tokenizer.tokenize(keyword))
    all_data.append(tokenizer.tokenize(keyword))
    transcription_data.append(word_tokenize(result))
    all_data.append(word_tokenize(result))
    new_keywords = []
    for i in word_tokenize(result):
      if i in tokenizer.tokenize(keyword):
        if not hasNumbers(i):
          new_keywords.append(1)
      else:
        new_keywords.append(0)
    if sum(new_keywords) != 0:
      sentence_column.append(word_tokenize(result))
      keyword_column.append(new_keywords)
    # get maximum length of words in a sentence
    if MAX_LEN < len(word_tokenize(result)):
      MAX_LEN = len(word_tokenize(result))

unique_words = getUniqueWords(all_data)

with open('dictionary.txt', 'w') as filehandle:
    for listitem in unique_words:
        filehandle.write('%s\n' % listitem)

# get vocab size
VOCAB_SIZE = len(unique_words)

words_tokenizer = Tokenizer(oov_token='__UNKNOWN__')
words_tokenizer.fit_on_texts(unique_words)
X = words_tokenizer.texts_to_sequences(sentence_column)
X = pad_sequences(X, padding = "post", truncating = "post", maxlen = MAX_LEN, value = 0)
y = pad_sequences(keyword_column, padding = "post", truncating = "post", maxlen = MAX_LEN, value = 0)
y = [to_categorical(i, num_classes = 2) for i in y]

embeddings_index = {}
f = open('glove.6B.100d.txt','r')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype = "float32")
	embeddings_index[word] = coefs
f.close()

ed = 100
word_index = words_tokenizer.word_index
word_index['__PADDING__'] = 0
embedding_matrix = np.zeros((len(word_index) + 1, ed))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
model = Sequential()
model.add(Embedding(len(word_index) + 1, 100, weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(128, return_sequences = True, recurrent_dropout = 0.1)))
model.add(TimeDistributed(Dense(2, activation = "softmax")))
model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X_train, y_train, batch_size = 32, epochs = 15, validation_split = 0.1)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")
model.save_weights("model_weight.h5")
pickle.dump(words_tokenizer, open("tokenizer.pickle","wb"))

