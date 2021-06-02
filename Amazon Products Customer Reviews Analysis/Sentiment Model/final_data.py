# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 01:05:43 2021

@author: Suren
"""
import pandas as pd
import pickle
sentiment_1,sentiment_2,sentiment_3=pickle.load(open('sentiment_test_model.pkl','rb'))


amazon_dataset_name_1 = 'data_1'
amazon_data_1 = pd.read_csv(amazon_dataset_name_1)

amazon_dataset_name_2 = 'data_2'
amazon_data_2 = pd.read_csv(amazon_dataset_name_2)
amazon_dataset_name_3 = 'data_3'
amazon_data_3 = pd.read_csv(amazon_dataset_name_3)   
    
amazon_data_1['Sentiment'] = sentiment_1
amazon_data_2['Sentiment'] = sentiment_2
amazon_data_3['Sentiment'] = sentiment_3

amazon_data_1.to_csv('data1.csv')
amazon_data_2.to_csv('data2.csv')
amazon_data_3.to_csv('data3.csv')
