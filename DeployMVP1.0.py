    
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re # for regex
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
import numpy as np
import itertools
import matplotlib.pyplot as plt

    
st.title("Análise de Sentimento de Feedbacks de Compras de Eletrónicos no Varejo - Case Amazon** ")
#st.subheader("Paras Patidar - MLAIT")
st.write('\n\n')
    


    
@st.cache
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
        return rem
def to_lower(text):
    return text.lower()
import nltk
nltk.download('stopwords')
nltk.download('punkt')
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]
def stem_txt(text):
    ss = SnowballStemmer('english') #tem portugues também
    return " ".join([ss.stem(w) for w in text])
def print_result(result):
    text,analysis_result = result
    print_text = "Positive" if analysis_result[0]=='1' else "Negative"
    return text,print_text

option = st.selectbox('Selecione o Modelo Que Deseja',('GB','BernoulliNB','Rede Neural','Regressão Logística'))
st.write('You selected:', option)
if option == 'GB':
    with open('gb_clf_sa.pkl','rb') as f:
        classifier =  pickle.load(f)
if option == 'BernoulliNB':
    with open('clf_bnb.pkl','rb') as f:
        classifier =  pickle.load(f)
if option == 'Rede Neural':
    from keras.models import model_from_json
    json_file = open('model13.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model13.h5")
    classifier = loaded_model
else:
    with open('clf_lr.pkl','rb') as f:
        classifier =  pickle.load(f)
review = st.text_input("Enter The Review","Write Here...")
if st.button('Predict Sentiment'):
    with open('cv.pkl','rb') as f:
        cv = pickle.load(f)
        f11 = clean(review)
        f22 = is_special(f11)
        f33 = to_lower(f22)
        f44 = rem_stopwords(f33)
        f55 = stem_txt(f44)
        bow,words = [],word_tokenize(f55)
        for word in words:
            bow.append(words.count(word))
        word_dict = cv.vocabulary_
        inp3 = []
        for i in word_dict:
            inp3.append(f55.count(i[0]))
        y_pred2 = classifier.predict(np.array(inp3).reshape(1,2000))
        if y_pred2 == 0:
            a = 'POSITIVO'
        else:
            a = 'NEGATIVO'
        st.success(a)
else:
    st.write("Press the above button..")





