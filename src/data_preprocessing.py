import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from logger import logging

# globals
porter_stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()
scaler = StandardScaler(with_mean=False)
STOPWORDS = set(stopwords.words("english"))


def stemming(content):
    content = re.sub("[^a-zA-Z]", " ", content)
    content = content.lower().split()
    content = [
        porter_stemmer.stem(word)
        for word in content
        if word not in STOPWORDS
    ]
    return " ".join(content)


def read_csv():
    logging.info("Reading dataset")
    return pd.read_csv("data/raw/data.csv")


def preprocess():
    df = read_csv()
    logging.info("Starting preprocessing")

    # numeric → categorical
    df["match_score_cls"] = df["match_score"].replace({
        1: "low",
        2: "Medium",
        3: "Medium",
        4: "High",
        5: "High"
    })

    df.drop(columns=["match_score"], inplace=True)

    oe = OrdinalEncoder(categories=[["low", "Medium", "High"]])
    df["match_score_cls"] = oe.fit_transform(
        df[["match_score_cls"]]
    ).astype(int)

    # text cleaning
    df["job_description"] = df["job_description"].apply(stemming)
    df["resume"] = df["resume"].apply(stemming)

    # TF-IDF (fit on combined corpus)
    corpus = df["job_description"].tolist() + df["resume"].tolist()
    vectorizer.fit(corpus)

    job_tfidf = vectorizer.transform(df["job_description"])
    resume_tfidf = vectorizer.transform(df["resume"])

    # cosine similarity
    df["similarity"] = [
        cosine_similarity(job_tfidf[i], resume_tfidf[i])[0][0]
        for i in range(len(df))
    ]

    # feature matrix
    X = np.hstack((
        job_tfidf.toarray(),
        resume_tfidf.toarray(),
        df["similarity"].values.reshape(-1, 1)
    ))

    y = df["match_score_cls"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info("Preprocessing completed")
    return X_train, X_test, y_train, y_test

