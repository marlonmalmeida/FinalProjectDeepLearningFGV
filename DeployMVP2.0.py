
import pickle
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

st.title("Análise de Sentimento de Feedbacks de Compras de Eletrónicos no Varejo - Case Amazon** ")
st.subheader("Arquivo CSV - Abaixo os Feedbacks Negativos")
st.write('\n\n')

@st.cache


@st.cache

def resumo_xval_score(model, X, Y):

  accuracy = cross_val_score(model, X, Y,  scoring="accuracy")
  precision = cross_val_score(model, X, Y, scoring="precision")
  recall = cross_val_score(model, X, Y,  scoring="recall")
  f1 = cross_val_score(model, X, Y, scoring="f1")

  return accuracy.mean(), precision.mean(), recall.mean(), f1.mean()

def print_result(result):
    text,analysis_result = result
    print_text = "Positive" if analysis_result[0]=='1' else "Negative"
    return text,print_text

#review = st.text_input("Enter The Review","Write Here...")
data = st.file_uploader('Escolha o dataset (.csv)', type = 'csv')
if data is not None:
    df = pd.read_csv(data)
    df.drop(df[df.iloc[:,2].isna()].index, inplace=True)
    df = df.rename(columns={'Review body': 'review_body','Reviewer name':'reviewer_name','Review rating':'review_rating'})
    df['Negativo'] = df['sentiment'].map( {'negative': 1, 'positive': 0} ).astype(int)
if st.button('Predict Sentiment'):
    with open('clf_bnb.pkl','rb') as f:
        modelpronto =  pickle.load(f)
        cv = CountVectorizer(max_features= 2000)
        review2 = cv.fit_transform(np.array(df.iloc[:,2].values))
        y_pred = modelpronto.predict(review2)
        for i in range(1,100):
            if y_pred[i]==1:
                df15 = ([i] + [df.review_body[i]])
                st.success(df15)
        score_gs_test_bnb = resumo_xval_score(modelpronto, review2, y_pred)
        score_gs_test_df = pd.DataFrame(np.array([score_gs_test_bnb]),index=['BN'],columns=['Accuracy', 'Precision', 'Recall', 'F1'])
        st.dataframe(score_gs_test_df) 
       # y_pred = modelpronto.predict(review2[1])
    #st.success(y_pred)
    #st.success(df["Review body"][0])
else:
    st.write("Press the above button..")
    


    


       