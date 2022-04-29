from flask import Flask, request
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']

    # result = model.predict(message)[0]
    # return jsonify({'result':str(result)})
    # processed_text = text.upper()
    # return processed_text

    tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
    model = pickle.load(open('./model.pkl', 'rb'))

    # 1. preprocess
    transformed_sms = transform_text(text)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        return 'spam';
    else:
        return 'ham';


@app.route('/sentiment', methods=['POST'])
def predict2():
    text = request.form['input']

    # result = model.predict(message)[0]
    # return jsonify({'result':str(result)})
    # processed_text = text.upper()
    # return processed_text

    tfidf = pickle.load(open('./tf_vect.pkl', 'rb'))
    model2 = pickle.load(open('./Sentiment_model.pkl', 'rb'))

    # 1. preprocess
    transformed_sms = transform_text(text)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result2 = model2.predict(vector_input)[0]
    # 4. Display
    if result2 == 1:
        return 'positive';
    else:
        return 'negative';


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if __name__ == '__main__':
    app.run(debug=True)
