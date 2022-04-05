
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
np.random.seed(42)

#pip install nltk

import requests as re
import re # for regex
import nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

df2 = pd.read_csv(r"C://Users//marlo//deepprojetos//amazonReview.csv")
df2.head()
df2.info()
df2.describe()
sentiment_prop = df2["sentiment"].value_counts()/len(df2)
sentiment_prop
df2["sentiment"].value_counts()
print(df2['sentiment'].isna().sum())
print(df2['Review body'].isna().sum())
#Dealing with missing values
df2.drop(df2[df2['Review body'].isna()].index, inplace=True)
df2['Review body'].isna().sum()

df3 = df2.reset_index()
df3.head()

#Rename the Features.
df3 = df3.rename(columns={'Review body': 'review_body','Reviewer name':'reviewer_name','Review rating':'review_rating'})
df3

#Encoding the "Negative Value" 
df3['Negativo'] = df3['sentiment'].map( {'negative': 1, 'positive': 0} ).astype(int)
df3

#Criamos data base com menor observação para analise preliminares dos dados 
df5 = df3.copy()
df5 = df5.sample(80000)
df5



from sklearn.feature_extraction.text import CountVectorizer

def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

df3.review_body = df3.review_body.apply(clean)
df3.review_body[48946]


def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

df3.review_body = df3.review_body.apply(is_special)
df3.review_body[0]



def to_lower(text):
    return text.lower()

df3.review_body = df3.review_body.apply(to_lower)
df3.review_body[0]


import nltk
nltk.download('stopwords')
nltk.download('punkt')

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

df3.review_body = df3.review_body.apply(rem_stopwords)
df3.review_body[0]

def stem_txt(text):
    ss = SnowballStemmer('english') #tem portugues também
    return " ".join([ss.stem(w) for w in text])

df3.review_body = df3.review_body.apply(stem_txt)
df3.review_body[0]

#Creat a new data frame withoud laybor
todrop = ['sentiment','Negativo']
df4 = df3.drop(todrop, axis=1)
df4.head()

#We found that the base is unbalanced with most positive ratings
sentiment_prop = df3.groupby('Negativo')['index'].count()/len(df3)
sentiment_prop

#Criamos data base com menor observação para analise preliminares de modelos 
df6 = df3.copy()
df6 = df6.sample(80000)
df6.reset_index(drop=True, inplace=True)
df6

del df6['index']
df6

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in split.split(df6, df6['Negativo']):
    train_set = df6.loc[train_index]
    test_set = df6.loc[test_index]

#Utilizaremos essa linha de código para validação na amostra completa e desabilitaremos código anterior
#from sklearn.model_selection import StratifiedShuffleSplit
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
#for train_index, test_index in split.split(df6, df6['Negativo']):
#    train_set = df6.loc[train_index]
#    test_set = df6.loc[test_index]

print(f"Treino: {len(train_set)}\nTeste: {len(test_set)}")

#Definindo o X e Y da base
train_setx = train_set.copy()
test_setx = test_set.copy()
xtrain = np.array(train_setx.drop('Negativo', axis=1, inplace=True))
xtest = np.array(test_setx.drop('Negativo', axis=1, inplace=True))

ytrain = np.array(train_set['Negativo'])
ytest=np.array(test_set['Negativo'])

Xtest_reg = np.array(test_setx.iloc[:,2].values)
Xtrain_reg = np.array(train_setx.iloc[:,2].values)
ytrain = np.array(train_set.Negativo.values)
ytest = np.array(test_set.Negativo.values)
cv = CountVectorizer(max_features = 1000)
Xtrain_reg = cv.fit_transform(train_setx.review_body).toarray()
Xtest_reg = cv.fit_transform(test_setx.review_body).toarray()
print("X.shape = ",Xtrain_reg.shape)
print("y.shape = ",ytrain.shape)
print("X.shape = ",Xtest_reg.shape)
print("y.shape = ",ytest.shape)


from imblearn.over_sampling import SMOTE
smt = SMOTE()

X1_train_smote = Xtrain_reg
Y1_train_smote = ytrain
X1_test_smote = Xtest_reg
Y1_test_smote= ytest
X1_train_smote, Y1_train_smote = smt.fit_resample(X1_train_smote, Y1_train_smote)
X1_test_smote, Y1_test_smote = smt.fit_resample(X1_test_smote, Y1_test_smote)

np.bincount(Y1_train_smote)
np.bincount(Y1_test_smote)

#import seaborn as sns
ax = sns.countplot(Y1_train_smote)

#import seaborn as sns
ax_test = sns.countplot(Y1_test_smote)


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


from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB().fit(X1_train_smote, Y1_train_smote)

from sklearn.naive_bayes import BernoulliNB
clf_bnb = BernoulliNB(alpha=0.20, 
                     binarize=0.0, 
                     fit_prior=True, 
                     class_prior=None).fit(X1_train_smote, Y1_train_smote)

#!pip install imblearn
#!pip install scikit-learn

#Modelo Regressão Linear
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
lr = LogisticRegression(max_iter=200)
clf_lr = lr.fit(X1_train_smote, Y1_train_smote)
pred_train_lr_smote = clf_lr.predict(X1_train_smote)
pred_train_lr_test_smote = clf_lr.predict(X1_test_smote)


from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
gb_clf_sa =gb_clf.fit(X1_train_smote, Y1_train_smote)
pred_train_gb_clf = gb_clf_sa.predict(X1_train_smote)
pred_test_gb_clf = gb_clf_sa.predict(X1_test_smote)


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model10 = keras.Sequential([layers.Dense(4, activation='relu'),
                          layers.Dense(4, activation='sigmoid'),
                          layers.Dense(1),
                         ])
opt = tf.keras.optimizers.Adam(learning_rate=0.0001,name='Adam')
model10.compile(optimizer=opt, loss='mae')
history10 = model10.fit(X1_train_smote, Y1_train_smote,
                    validation_data=(X1_test_smote, Y1_test_smote),
                    batch_size=64,
                    epochs=50,
                    verbose=0,  # turn off training log
                   )

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(min_delta=0.001, # minimium amount of change to count as an improvement
                               patience=20, # how many epochs to wait before stopping
                               restore_best_weights=True,
                              )

model13 = keras.Sequential([layers.Dense(12, activation='relu'),
                            layers.Dropout(0.3),
                            layers.Dense(8, activation='relu'),
                            layers.Dense(8, activation='sigmoid'),
                            layers.Dense(1),
                         ])
opt = tf.keras.optimizers.Adam(learning_rate=0.0001,name='Adam')
model13.compile(optimizer=opt, loss='mae',metrics=['binary_accuracy'])
history13 = model13.fit(X1_train_smote, Y1_train_smote,
                    validation_data=(X1_test_smote, Y1_test_smote),
                    batch_size=64,
                    epochs=50,
                    callbacks=[early_stopping],
                    verbose=1,  # turn off training log
                   )

history11 = model10.fit(X1_train_smote, Y1_train_smote)
def transformar(rev2):
    f11 = clean(rev2)
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
        y_pred2 = gb_clf_sa.predict(np.array(inp3).reshape(1,1000))
trans = transformar
#Salvando modelo 
with open ('gb_clf_sa.pkl','wb') as f:
    pickle.dump(gb_clf_sa, f)
with open('cv.pkl','wb') as f:
    pickle.dump(cv,f)
with open('trans.pkl','wb') as f:
    pickle.dump(trans,f)
with open('clf_lr.pkl','wb') as f:
    pickle.dump(clf_lr,f)
with open('clf_bnb.pkl','wb') as f:
    pickle.dump(clf_lr,f)

# serialize model to JSON
from keras.models import model_from_json
# serialize model to JSON
model13_json = model13.to_json()
with open("model13.json", "w") as json_file:
    json_file.write(model13_json)
# serialize weights to HDF5
model13.save_weights("model13.h5")
print("Saved model to disk")



