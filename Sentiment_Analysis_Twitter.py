# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:25:53 2019

@author: ritap
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Embedding, LSTM, Dropout, Activation
from sklearn.model_selection import train_test_split

def use_gpu():
    # Creates a graph.
    with tf.device('/gpu:0'):
      a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
      b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))
    print("Using The GPU")

use_gpu()

data = pd.read_csv('train_2kmZucJ.csv', index_col =0)
sen = data.loc[:,'tweet']
labels = data.loc[:,'label']
train_sen, test_sen, train_labels, test_labels = train_test_split(sen, labels, test_size=0.2, random_state=42)
 
t = Tokenizer()
t.fit_on_texts(train_sen)
vocab_size = len(t.word_index) + 1

encoded_train_sen = t.texts_to_sequences(train_sen)
print(encoded_train_sen)
encoded_test_sen = t.texts_to_sequences(test_sen)
print(encoded_test_sen)

max_length = 50
padded_train_sen = pad_sequences(encoded_train_sen, maxlen=max_length, padding='post')
print(padded_train_sen)
padded_test_sen = pad_sequences(encoded_test_sen, maxlen=max_length, padding='post')
print(padded_test_sen)

embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
    
sentences = Input((50,), dtype='int32')
embeddings = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=50, trainable=False)(sentences)
X = LSTM(50, return_sequences=True)(embeddings)
X = Dropout(0.5)(X)   
X = LSTM(50, return_sequences=False)(X)
X = Dropout(0.5)(X)
X = Dense(1)(X)
X = Activation('sigmoid')(X)

model = Model(inputs=sentences, outputs=X)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_train_sen, train_labels, epochs=50, verbose=1, batch_size=32)   



X2 = LSTM(50, return_sequences=False)(embeddings)
X2 = Dropout(0.5)(X2)
X2 = Dense(1)(X2)
X2 = Activation('sigmoid')(X2)

model = Model(inputs=sentences, outputs=X)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_train_sen, train_labels, epochs=50, verbose=1, batch_size=32)

loss, accuracy = model.evaluate(padded_test_sen, test_labels, verbose=1)
print('Accuracy: %f' % (accuracy*100))
