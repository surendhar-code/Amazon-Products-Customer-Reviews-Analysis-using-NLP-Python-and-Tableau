# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 23:37:21 2021

@author: Suren
"""

import pandas as pd
from bs4 import BeautifulSoup 
import re
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

amazon_dataset_name_1 = 'data_1'
amazon_data_1 = pd.read_csv(amazon_dataset_name_1)

amazon_dataset_name_2 = 'data_2'
amazon_data_2 = pd.read_csv(amazon_dataset_name_2)
amazon_dataset_name_3 = 'data_3'
amazon_data_3 = pd.read_csv(amazon_dataset_name_3)   
    
    
    
 
sentences_1=[]
for i in range(0,len(amazon_data_1['Reviews'])):
    sentences_1.append(amazon_data_1['Reviews'][i])
    
sentences_2=[]
for i in range(0,len(amazon_data_2['Reviews'])):
    sentences_2.append(amazon_data_2['Reviews'][i])
    
sentences_3=[]
for i in range(0,len(amazon_data_3['Reviews'])):
    sentences_3.append(amazon_data_3['Reviews'][i])
    
def txt_preprocess(sentences):
  
   final_sent=[]
   print("Pre-processing started....")
   

   for i in range(0,len(sentences)):
 
    review_text=re.sub('[^a-zA-Z]',' ',sentences[i])    #to remove punctuation marks
    review_text=review_text.lower().split()            #lowering the sentence and word tokenization
    words=[word for word in review_text if word not in set(stopwords.words('english'))] #removing stopwords
    review_text=' '.join(words) #joining the word after pre-processing
    final_sent.append(review_text)
   
   print("pre-processing finished...") 
  
   return final_sent

final_sent_1 = txt_preprocess(sentences_1)
final_sent_2 = txt_preprocess(sentences_2)
final_sent_3 = txt_preprocess(sentences_3)



import pickle
pickle.dump([final_sent_1,final_sent_2,final_sent_3],open('txt_preprocessing.pkl','wb'))

