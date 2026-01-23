from .data_preprocessing import preprocess
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from .logger import logging

import mlflow
import mlflow.sklearn
import dagshub
import os
from dotenv import load_dotenv


load_dotenv()
#dagshub initilization : 
dagshub_token=os.getenv("ENV_VAR")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner="seika-afk"
repo_name="Smart-Resume-Screener"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def train_model():
    x_train,x_test,y_train,y_test=preprocess()
    model=RandomForestClassifier()
    model.fit(x_train,y_train)
    
    #joblib.dump(model,'../models/model.pkl')
    return model,x_test,y_test


def evaluate(y_test,y_preds):
    acc=accuracy_score(y_test,y_preds)
    return acc
def main():
    with mlflow.start_run():  
        model, x_test, y_test = train_model()
        logging.info("Model Trained.")
        
        preds = model.predict(x_test)
        acc = evaluate(y_test, preds)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        logging.info("Tracked model")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        logging.info("Tracked metric")
        logging.info("Finished")


if __name__ == "__main__":
    main()

