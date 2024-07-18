import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify
import random

nltk.download('movie_reviews')

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

texts = [" ".join(doc) for doc, _ in documents]
labels = [label for _, label in documents]

vectorizer =  CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(texts)

X_train , X_test, y_train, y_test = train_test_split(X, labels,test_size=0.25, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/home', methods=['GET'])
def index():
    return "Hello World!"


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    text = json_data['text']
    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)
    return jsonify({'prediction': prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)