# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB




data = 'pre-data'
pre_data = pd.read_csv(data)

X=pre_data['Reviews']
Y=pre_data['Sentiment']

#train and test dataset
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
tf_idf=TfidfVectorizer(max_features=10000)
x_train_tf=tf_idf.fit_transform(x_train)
   
mul_nb=MultinomialNB()
mul_nb.fit(x_train_tf,y_train)

import pickle
pickle.dump([tf_idf, mul_nb], open('sentiment_train.pkl', 'wb' ))








        