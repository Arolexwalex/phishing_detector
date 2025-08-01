from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from flask import Flask, request, render_template, session
import zipfile
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

# Unzip dataset with memory consideration (load only necessary data)
with zipfile.ZipFile('phishing_email.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Load a sample of the dataset to reduce memory
data = pd.read_csv('phishing_email.csv', nrows=10000)  # Limit to 10,000 rows
emails = data['text_combined'].fillna('')
labels = data['label']

# Feature extraction with n-grams
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)  # Reduce features
X = vectorizer.fit_transform(emails)
model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduce estimators
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
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default
    app.run(host='0.0.0.0', port=port, debug=True)