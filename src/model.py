# -*-coding:Utf-8 -


"""
This script trains a sentiment classifier using a Naive Bayes model with text preprocessing techniques.
It reads labeled movie review data, pre-processes the text, and trains a Multinomial Naive Bayes classifier.
The model performance is evaluated using accuracy score and the trained classifier is saved to disk.

@Date: April 26th, 2023
@Author: Yassine RODANI
"""

import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

# dependencies
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
    "punkt"
):
    nltk.download(dependency)

warnings.filterwarnings("ignore")

# stopwords
nltk.download('stopwords')
stop_words = stopwords.words("english")

# seeding
np.random.seed(123)


def process_review(text, remove_stop_words=True, lemmatize_words=True, stemming_words=True) -> str:
    """
    Clean and preprocess text data by removing unwanted characters, stop words, and lemmatizing words.

    This function takes a raw text input and processes it by removing non-alphanumeric characters,
    punctuation, numbers, and optionally, stop words, lemmatizing and stemming. The output is a cleaned
    and preprocessed version of the input text.

    Parameters:
    text (str): The raw text input to be processed.
    remove_stop_words (bool, optional): If True, removes stop words from the input text. Defaults to True.
    lemmatize_words (bool, optional): If True, lemmatizes words in the input text. Defaults to True.
    stemming_words (bool, optional): If True, lemmatizes words in the input text. Defaults to True.

    Returns:
    str: The cleaned and preprocessed version of the input text.
    """

    # Remove non-alphanumeric characters, punctuation, and numbers
    text = re.sub(r'\W+', ' ', text.lower())

    # Remove hyperlinks
    text = re.sub(r'http\S+|www\S+', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    if remove_stop_words:
        tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize words
    if lemmatize_words:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stem words
    if stemming_words:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)


def main() -> None :
    # load data
    data = pd.read_csv("data/labeledTrainData.tsv", sep='\t')
    # clean the review
    data["cleaned_review"] = data["review"].apply(process_review)

    # split features and target from  data
    X = data["cleaned_review"]
    y = data.sentiment.values

    # split data into train and validate
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

    # create a classifier in pipeline
    sentiment_classifier = Pipeline(steps=[
        ('pre_processing', TfidfVectorizer(lowercase=False)),
        ('naive_bayes', MultinomialNB())
    ])

    # train the sentiment classifier
    sentiment_classifier.fit(X_train, y_train)

    # test model performance on valid data
    y_preds = sentiment_classifier.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_preds)
    print(f"Accuracy: {accuracy:.4f}")

    # save model
    joblib.dump(sentiment_classifier, 'models/sentiment_model_pipeline.pkl')


if __name__ == "__main__":
    main()
