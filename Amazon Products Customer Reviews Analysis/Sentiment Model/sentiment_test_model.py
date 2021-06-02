# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 23:37:52 2021

@author: Suren
"""
import pickle
tf_idf,mul_nb=pickle.load(open('sentiment_train.pkl','rb'))
final_sent_1,final_sent_2,final_sent_3=pickle.load(open('txt_preprocessing.pkl','rb'))




def sentiment_test_model(sent,tf_idf,model):
    print("started...........")
    user_ip_tf=tf_idf.transform(sent).toarray()
    pred=model.predict(user_ip_tf)
    pred.tolist()
    print("finished............")
    return pred

sentiment_1 = sentiment_test_model(final_sent_1,tf_idf,mul_nb)
print("The length of sentiment_1 : ",len(sentiment_1))

sentiment_2 = sentiment_test_model(final_sent_2,tf_idf,mul_nb)
print("The length of sentiment_2 : ",len(sentiment_2))

sentiment_3 = sentiment_test_model(final_sent_3,tf_idf,mul_nb)
print("The length of sentiment_3 : ",len(sentiment_3))




pickle.dump([sentiment_1,sentiment_2,sentiment_3],open('sentiment_test_model.pkl','wb'))




