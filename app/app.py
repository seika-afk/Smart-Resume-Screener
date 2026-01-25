from flask import Flask, render_template, request
import os
import re
import numpy as np
import pdfplumber
import joblib

from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

porter_stemmer = PorterStemmer()


def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [
        porter_stemmer.stem(word)
        for word in content
        if word not in stopwords.words('english')
    ]
    return " ".join(content)


def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def evaluate_resume(jd_text, resume_path):
    resume_text = extract_pdf_text(resume_path)
    x_resume = stemming(resume_text)
    x_jd = stemming(jd_text)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

    jd_vec = vectorizer.transform([x_jd])
    resume_vec = vectorizer.transform([x_resume])

    similarity = cosine_similarity(jd_vec, resume_vec)[0][0]

    X = np.hstack((
        jd_vec.toarray(),
        resume_vec.toarray(),
        np.array([[similarity]])
    ))

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]

    if pred > 1:
        verdict = "Good Match"
        pred= "High"
    elif pred == 1:
        verdict = "Decently Matches"
        pred="Medium"
    else:
        verdict = "Needs Improvement"
        pred="Low"
    return {
        "score": pred,
        "verdict": verdict
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd_text = request.form.get("jd")

        resume = request.files.get("resume")
        resume_path = None

        if resume and resume.filename:
            filename = secure_filename(resume.filename)
            resume_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            resume.save(resume_path)

        result = evaluate_resume(jd_text, resume_path)

        return render_template(
            "result.html",
            score=result["score"],
            verdict=result["verdict"]
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
