# -*-coding:Utf-8 -


"""
A simple machine learning application to predict the sentiment of a movie review.

This script loads a trained sentiment classification model and uses it to predict
the sentiment of a movie review entered by the user. The user interface is built using
Streamlit, and includes a form to receive the review text and a button to trigger the
prediction.

The script uses text preprocessing techniques, such as removing stop words and lemmatizing
words, to clean the input text before making the prediction. The cleaned text is then passed
to the model, which predicts whether the sentiment of the review is positive or negative.
The predicted sentiment and its probability are then displayed to the user.

@Date: April 16th, 2023
@Author: Yassine RODANI
"""

# import packages
import streamlit as st
from trubrics.integrations.streamlit import FeedbackCollector
import datetime
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re  # regular expression
import joblib

import warnings

warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

# load stop words
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words("english")


# function to clean the text

@st.cache
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
    lemmatize_words (bool, optional): If True, lemmatizes words in the input text. Defaults to True.

    Returns:
    str: The cleaned and preprocessed version of the input text.
    """

    # clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'https?://[^\s]+', '', text)  # remove hyperlinks
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers

    # remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # lemmatize words
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # shorten words to their stems
    if stemming_words:
        text = text.split()
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text

# functon to make prediction


@st.cache
def make_prediction(review)-> tuple:
    """
    Predicts the sentiment of a movie review using a trained classification model.

    This function takes a movie review text as input, cleans the text using the
    `process_review` function, loads a trained sentiment classification model from
    a saved file, and uses the model to predict the sentiment of the review. The
    predicted sentiment and its probability are returned as a tuple.

    Parameters:
        review (str): The movie review text to be classified.

    Returns:
        tuple: A tuple containing the predicted sentiment (1 for positive, 0 for negative)
               and its probability (a float between 0 and 1).
    """

    # clearn the data
    clean_review = process_review(review)

    # load the model and make prediction
    model = joblib.load("models/sentiment_model_pipeline.pkl")

    # make prection
    result = model.predict([clean_review])

    # check probabilities
    probas = model.predict_proba([clean_review])
    probability = "{:.2f}".format(float(probas[:, result]))

    return result, probability


# Set page configuration
st.set_page_config(page_title="Movie Sentiment Analysis", layout="centered", page_icon="🎬")
st.title("Movie Sentiment Analysis")
st.write(
    "A simple machine learning application to predict the sentiment of a movie's review"
)

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

# Set page sidebar
with st.sidebar:
    st.markdown("""
    # About 
    By leveraging the power of advanced natural language processing techniques, 
    this simple app is designed to accurately predict the sentiment of a movie review, 
    providing a quick and efficient way to gauge public opinion on films.
    
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),
                unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Simply input the text of your movie's review in the text box below and click on the 'Make Prediction' button.
    
    For instance, you can try the following review:
    
    "WOW!! Elysian Dreams is a mesmerizing cinematic masterpiece, delivering a 
    powerful narrative, outstanding performances, and captivating visuals that 
    leave audiences spellbound."
    
    
    
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),
                unsafe_allow_html=True)
    st.markdown("""
    Made by [yassine-rd](https://github.com/yassine-rd) with ❤️
    """,
                unsafe_allow_html=True,
                )

# Include a form to receive a movie's review
form = st.form(key="my_form")
review = form.text_input(label="Please enter the text of your movie's review")
submit = form.form_submit_button(label="Make Prediction")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.probability = None

if submit:
    # Make prediction from the input text
    result, probability = make_prediction(review)

    # Save the results to session state
    st.session_state.result = result
    st.session_state.probability = probability

if st.session_state.result is not None:
    # Display results of the NLP task
    st.header("Results")

    if int(st.session_state.result) == 1:
        st.write("This is a positive review with a probability of ", st.session_state.probability)
    else:
        st.write("This is a negative review with a probability of ", st.session_state.probability)

    st.subheader("How satisfied are you with this prediction?")
    collector = FeedbackCollector()
    collector.st_feedback(feedback_type="faces",
                          path=f"feeds/thumbs_{datetime.datetime.now().timestamp()}.json",
                          open_feedback_label="An open text field")
