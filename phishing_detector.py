from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask import Flask, request, render_template, session
import zipfile

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

# Unzip dataset
with zipfile.ZipFile('phishing_email.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load dataset
data = pd.read_csv('phishing_email.csv')
emails = data['text_combined'].fillna('')
labels = data['label']

# Feature extraction with n-grams
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000)
X = vectorizer.fit_transform(emails)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, labels)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = session.get('result')
    if request.method == 'POST':
        email_text = request.form['email']
        X_new = vectorizer.transform([email_text])
        result = "Phishing Alert!" if model.predict(X_new)[0] == 1 else "Safe Email"
        session['result'] = result
    else:
        session.pop('result', None)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)