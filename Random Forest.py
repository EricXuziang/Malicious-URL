#!/usr/bin/env python
# coding: utf-8

# In[4]:


# coding: utf-8
# import Pandas library to load data
import pandas as pd
import numpy as np
import random
# import Sklearn library 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Load url data 
urls_data = pd.read_csv("data1.csv")


# In[6]:


# View type of url
type(urls_data)


# In[7]:


urls_data.head()


# In[48]:


# another method about TFidf
'''
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/') # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')	# make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')	# make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))	#remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens


# In[8]:


# store url
urls = urls_data["url"]
# store type
y = urls_data["type"]


# In[9]:


# Using ngram algorithm for word split
def nGram(a):
    temp=str(a)
    Gram=[]
    for i in range(0,len(temp)-3):
        Gram.append(temp[i:i+3])
    return Gram


# In[10]:


# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=nGram)
# Store vectors into X variable as Our X Features
X = vectorizer.fit_transform(urls)
print(X)


# In[11]:


# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


# Model Building
# Using randomforest and default parameter
random=RandomForestClassifier()
random.fit(X_train,y_train)


# In[13]:


print("Accuracy ",random.score(X_test, y_test))


# In[35]:


# Visual view has not been implemented 
'''
import time
import matplotlib.pyplot as plt
from typing import List

test_history_acc: List[float] = []
test_history_loss: List[float] = []


class Logger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_history_acc.append(log.get("acc"))
        test_history_acc.append(log.get("loss"))


# train_data = pd.read_csv("banknote_train_data.csv")
# test_data = pd.read_csv("banknote_test_data.csv")


train_data = pd.read_csv("URL_training_data.csv")
test_data = pd.read_csv("URL_training_data.csv")

train_data_raw = train_data.values
inputs_train = train_data_raw[:, :4]
labels_train = train_data_raw[:, 4:]

test_data_raw = test_data.values
inputs_test = test_data_raw[:, :4]
labels_test = test_data_raw[:, 4:]


model = K.models.Sequential()
model.add(K.layers.Dense(units=8, input_dim=4,
                         activation='relu', kernel_initializer="he_normal"))
model.add(K.layers.Dense(units=8, activation='relu',
                         kernel_initializer="he_normal"))
model.add(K.layers.Dense(units=1, activation='sigmoid',
                         kernel_initializer="glorot_uniform"))

adam_optimiser = K.optimizers.Adam(
    lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**-8)
model.compile(loss='binary_crossentropy',
              optimizer=adam_optimiser, metrics=['accuracy'])

# https://keras.io/visualization/

num_epochs = 50
batch_size = 32
start = time.time()
history = model.fit(x=inputs_train, y=labels_train, epochs=num_epochs, batch_size=batch_size,
                    validation_split=0, validation_data=(inputs_test, labels_test), verbose=0)
training_and_validation_done = time.time()
eval_result = model.evaluate(x=inputs_test, y=labels_test, verbose=0)

plt.figure(0)
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Malicious URL Identifier Accuracy - MLP")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy", "Testing Accuracy"], loc="lower right")

plt.figure(1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Malicious URL Identifier Loss - MLP")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training Loss", "Testing Loss"], loc="upper right")
print("\nTesting Dataset on Trained Model -")
print("Accuracy: %f \nLoss: %f" % (eval_result[1], eval_result[0]))
print("Execution time (train and validate): %f secs" %
      (training_and_validation_done - start))

print(model.predict(inputs_test))
plt.show()


# In[13]:


# Use the gridsearchcv method to select the optimal parameters
from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(10,100,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=4,
                                                           min_samples_leaf=1,
                                                           random_state=42,max_depth=100),
                        param_grid = param_test1,cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[18]:


param_test2 = {'min_samples_split':range(2,11,2),'min_samples_leaf':range(1,10,2)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 50,random_state=42),
                         param_grid = param_test2,cv=5)
gsearch2.fit(X_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[20]:


param_test3 = {'max_depth':range(10,120,20)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 50,min_samples_split=4,
                                                           min_samples_leaf=1,random_state=42),
                         param_grid = param_test3,cv=5)
gsearch3.fit(X_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[22]:


# Use new parameters to test accuracy
rf1 = RandomForestClassifier(n_estimators=90,min_samples_split=6,
                                  min_samples_leaf=1,max_depth=110,criterion='gini')
rf1.fit(X_train, y_train)
print("Accuracy ",rf1.score(X_test, y_test))

