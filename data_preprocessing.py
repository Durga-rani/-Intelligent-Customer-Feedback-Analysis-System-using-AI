import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('customer_feedback.csv')
df.drop_duplicates(inplace=True)
df['feedback'] = df['feedback'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df.dropna(subset=['feedback'], inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_feedback'] = df['feedback'].apply(preprocess)


df.to_csv('cleaned_feedback.csv', index=False)
