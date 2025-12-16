import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import PyPDF2
import docx
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def summarize_text(text, jumlah):
    sentences = sent_tokenize(text)
    if len(sentences) <= jumlah:
        return text

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    scores = np.sum(tfidf.toarray(), axis=1)

    top_idx = np.argsort(scores)[-jumlah:]
    summary = " ".join([sentences[i] for i in sorted(top_idx)])
    return summary

def read_pdf(path):
    text = ""
    reader = PyPDF2.PdfReader(path)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def read_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs])

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        jumlah = int(request.form.get("jumlah", 3))
        text = request.form.get("text", "")

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            if file.filename.endswith(".pdf"):
                text = read_pdf(filepath)
            elif file.filename.endswith(".docx"):
                text = read_docx(filepath)

        if text.strip():
            summary = summarize_text(text, jumlah)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
