# Movie Sentiment Analysis: A Multinomial Na ̈ıve Bayes-Based Approach for Assessing User and Critic Opinions

## ✨ Abstract

The growing volume of user-generated content, particu- larly in the form of movie reviews, presents both challenges and opportunities for researchers and practitioners in the movie industry. Movie sentiment analysis has emerged as a vital tool to automatically process and interpret this vast repository of data, offering valuable insights into viewer preferences and opinions.

## 🎬 About the Project

In this project, we involve a comprehensive study of relevant NLP techniques, including data pre-processing, feature extraction, and model selection.

The chosen Multinomial Naïve Bayes algorithm will be trained and optimized on a dataset of user critic reviews, with model performance evaluated based on multiple evaluation metrics.

Deployed [here](https://movie-sentiment.streamlit.app/).

## 📁 Project Structure

The project consists of the following files:

- [app.py](app.py): The main file of the project. It loads a trained sentiment classification model and uses it to predict
the sentiment of a movie review entered by the user.
- [Movie_Sentiment_Analysis.ipynb](Movie_Sentiment_Analysis.ipynb): Contains the code for preprocessing the data, training the model, and evaluating the model.
- [data](./data/): Contains the dataset used for this project.
- [model](./model/): Contains the trained model.

## 💻 Try it out

### 🐳 Locally

A Docker image of the application has been deployed on Docker Hub. To run it locally, you need to have Docker installed on your machine. Then, run the following command:

```bash
docker pull yassine-rd/movie-sentiment-analysis
docker run -p 8501:8501 yassine-rd/movie-sentiment-analysis
```

### ☁️ On Streamlit Sharing

You can also run the app on Streamlit Sharing by clicking on the button below:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-sentiment.streamlit.app/)

## 💬 Contact

Reach out to [@yassine_rd_](https://twitter.com/yassine_rd_) on Twitter or feel free to contact me at: yassine.rodani@gmail.com
