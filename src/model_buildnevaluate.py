from .data_preprocessing import preprocess
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from .logger import logging

def train_model():
    x_train,x_test,y_train,y_test=preprocess()
    model=RandomForestClassifier()
    model.fit(x_train,y_train)
    
    #joblib,dump(model,'../models/model.pkl')
    return model,x_test,y_test


def evaluate(y_test,y_preds):
    acc=accuracy_score(y_test,y_preds)
    return acc


def main():
    model,x_test,y_test=train_model()
    logging.info("Model Trained.")
    preds=model.predict(x_test)
    acc=evaluate(y_test,preds)
    logging.info("Calculated Accuracy : %.4f",acc)
if __name__ == "__main__":
    main()

