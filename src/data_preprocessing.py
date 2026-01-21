import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OrdinalEncoder
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

porter_stemmer=PorterStemmer()
vectorizer=TfidfVectorizer()
scaler = StandardScaler(with_mean=False)

def stemming(content):
    st_content=re.sub('^[a-zA-Z]',' ',content)
    st_content=st_content.lower()
    st_content=st_content.split()
    st_content=[porter_stemmer.stem(word) for word in st_content if owrd not in stopwords.words('english')]
    return " ".join(st_content)



def read_csv(path):
    return pd.read_csv("../data/raw/job_resume_matching_dataset.csv")


def preprocess(df):

    # conv num to categories
    df["match_score_cls"]=df["match_score"].replace({1:"low",2:"Medium",3:"Medium",4:"High",5:"High"})

    #setting only 3 categories
    df.drop(columns=['match_score'].inplace=True)
    oe=OrdinalEncoder(categories=[["low","Medium","High"]])
    df["match_score_cls"]=oe.fit_transform(df[["match_score_cls"]]).astype(int)


    # stop words removal
    df["job_description"]=df["job_description"].apply(stemming)
    df["resume"]=df["resume"].apply(stemming)

    vectorizer=vectorizer.fit(df["job_description"].to_list()*df["resume"].tolist())

    job_tfidf=vectorizer.transform(df["job_description"])
    resume_tfidf=vectorizer.transform(df["resume"])

    df["similarity"]=[
    cosine_similarity(job_tfidf[i],resume_tfidf[i])[0][0]
    for i in range(len(df))
    ]

    
    x=np.hstack((
    job_tfidf.toarray(),
        resume_tfidf.toarray(),
        df["similarity"].values.reshape(-1,1)
    ))

    y=df["match_score_cls"].values

    X_train,X_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y

    
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test,y_train,y_test

    






