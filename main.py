import pandas as pd
from time import sleep as sleep

#Part1 : lOADING DATASETS
fake_df = pd.read_csv("false_dataset.csv")
true_df = pd.read_csv("true_dataset.csv")

#creating a new label column
true_df['label'] = 1
fake_df['label'] = 0

#merging the datasets
df = pd.concat([true_df,fake_df] , ignore_index=True)

#Part2 : PROCESSING DATASETS
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = re.sub(r'[^\w\s]' , '' , text) #removing punctuations
    tokens = word_tokenize(text.lower()) #tokenize and convert to lowercase
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)
print("-Data Processing Complete-")

#Part 3 : FEATURE EXTRACTION

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['clean_text'])
y = df['label']

print("-Feature Extraction Complete-")
sleep(2)

print("Preparing dependencies....")
sleep(1)
print("Initiating Model Training...")
sleep(1)

#Part 4 : MODEL SELECTION AND TRAINING
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sleep(1)
print("Model Training Algorithm Successfully INITIATED")
print("******************")
sleep(1)

print("Starting Training...")

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# --> IMPORTANT : Using Logistic Regression as algorithm for Binary Classification
model = LogisticRegression()
model.fit(X_train , y_train)

print("******************")
sleep(3)
print("MODEL TRAINED SUCCESFULLY")
print("******************")
sleep(1)
# Part 5: MODEL EVALUATION AND FINE TUNING
y_pred = model.predict(X_test)
accuracy= accuracy_score(y_test , y_pred)
print("Accuracy: " , accuracy)
# HIGHEST ACCURACY LEVEL : 98.697% (percentage)