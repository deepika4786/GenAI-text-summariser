from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

nltk.download('punkt')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        text = request.form['text']
        method = request.form['method']

        if method == 'extractive':
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
            result = summarizer(text, max_length=130, min_length=30, do_sample=False)
            summary = result[0]['summary_text']

        elif method == 'abstractive':
            model_name = "google/pegasus-xsum"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
            summary_ids = model.generate(inputs["input_ids"], max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
