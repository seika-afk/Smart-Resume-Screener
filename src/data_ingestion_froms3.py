import boto3
import pandas as pd
from logger import logging

from io import StringIO


class s3_operations:
    def __init__(self,bucket_name,aws_access_key,aws_secret_key,region_name="us-east-1"):
        self.bucket_name=bucket_name
        self.s3_client=boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name

                )
        logging.info("Data ingestion from s3 bucket initialized")
    def fetch_file_from_s3(self,file_key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
        df=pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        logging.info("Succefully fetched data")


data_ingestion=s3_operations("","AKIXXX","bo8pCtXXXXXXXXXhxnQ")
data_ingestion.fetch_file_from_s3("data.csv")
print("completed")
