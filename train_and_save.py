# train_and_save.py (updated)
import re
import os
import joblib
import sqlite3
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# ----------- config ----------
ps = PorterStemmer()
stopwords = set(ENGLISH_STOP_WORDS)
stopwords.discard('not')

# ---------- Build dataset ----------
reviews = []
labels = []

# Option 1: Use existing TSV
if os.path.exists('Restaurant_Reviews.tsv'):
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in stopwords]
        reviews.append(' '.join(review))
        labels.append(dataset.iloc[i, -1])

# Option 2: Include customer-submitted reviews from DB
if os.path.exists('Restaurant_food_data.db'):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    c.execute("SELECT item_name, no_of_positives, no_of_negatives FROM item")
    rows = c.fetchall()
    conn.close()

    for item, pos, neg in rows:
        pos, neg = int(pos), int(neg)
        reviews.extend([f"{item} was good"] * pos)
        labels.extend([1] * pos)
        reviews.extend([f"{item} was bad"] * neg)
        labels.extend([0] * neg)

# ----------- TF-IDF ----------
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(reviews).toarray()
y = labels

# ----------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ----------- Train Models ----------
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)

print("Training SVM...")
svm = LinearSVC(random_state=0, max_iter=10000)
svm.fit(X_train, y_train)

print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)

# ----------- Save Pickles ----------
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(rf, 'model_rf.pkl')
joblib.dump(svm, 'model_svm.pkl')
joblib.dump(nb, 'model_nb.pkl')

print("Saved tfidf.pkl and model_{rf,svm,nb}.pkl")

# Optional: print accuracy
from sklearn.metrics import accuracy_score
for name, model in [('Random Forest', rf), ('SVM', svm), ('Naive Bayes', nb)]:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")
