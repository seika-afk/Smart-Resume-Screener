FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

COPY app/ ./app/

COPY models/ ./models/

RUN mkdir -p uploads

EXPOSE 5000


CMD ["python", "app.py"]  

