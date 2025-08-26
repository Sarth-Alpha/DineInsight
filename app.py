# app.py  (updated)
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import re
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------- config ----------
FOODS = ["Idly", "Dosa", "Vada", "Roti", "Meals", "Veg Biryani",
         "Egg Biryani", "Chicken Biryani", "Mutton Biryani",
         "Ice Cream", "Noodles", "Manchooriya", "Orange juice",
         "Apple Juice", "Pineapple juice", "Banana juice"]

ps = PorterStemmer()
stopwords = set(ENGLISH_STOP_WORDS)
stopwords.discard('not')  # keep negation

# ---------- helper: load pretrained artifacts ----------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts['tfidf'] = joblib.load('tfidf.pkl')
        artifacts['model_nb'] = joblib.load('model_nb.pkl')
        artifacts['model_rf'] = joblib.load('model_rf.pkl')
        artifacts['model_svm'] = joblib.load('model_svm.pkl')
        artifacts['loaded'] = True
    except Exception:
        artifacts['loaded'] = False
        artifacts['tfidf'] = TfidfVectorizer(max_features=2000)
    return artifacts

art = load_artifacts()
tfidf = art['tfidf']

# If pickles are present, we'll use them. Otherwise the old training path will be used.
USE_PRETRAINED = art.get('loaded', False)

# ========== Database init (keeps original sqlite behavior) ==========
def init_data():
    if not os.path.exists('Restaurant_food_data.db'):
        conn = sqlite3.connect('Restaurant_food_data.db')
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS item (
                        item_name TEXT,
                        no_of_customers INTEGER,
                        no_of_positives INTEGER,
                        no_of_negatives INTEGER,
                        pos_perc TEXT,
                        neg_perc TEXT
                    )""")
        for food in FOODS:
            c.execute("INSERT INTO item VALUES (?,?,?,?,?,?)",
                      (food, 0, 0, 0, "0.0%", "0.0%"))
        conn.commit()
        conn.close()

init_data()

# ========== If pickles are NOT present, train on the fly (small dataset only) ==========
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

@st.cache_resource
def train_models_onfly():
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    corpus = []
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in stopwords]
        corpus.append(' '.join(review))

    X = tfidf.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    nb = GaussianNB()
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    svm = LinearSVC(random_state=0, max_iter=10000)

    nb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    # optional: evaluate and return
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {'nb': nb, 'rf': rf, 'svm': svm, 'acc': acc}

if not USE_PRETRAINED:
    models_pkg = train_models_onfly()
else:
    models_pkg = None

# --------- Model selector (uses pre-trained if present) ----------
model_choice = st.sidebar.selectbox("🔍 Choose ML Model", ["Naive Bayes", "Random Forest", "SVM"])

def get_classifier(choice):
    if USE_PRETRAINED:
        if choice == "Naive Bayes":
            return joblib.load('model_nb.pkl')
        if choice == "Random Forest":
            return joblib.load('model_rf.pkl')
        if choice == "SVM":
            return joblib.load('model_svm.pkl')
    else:
        if choice == "Naive Bayes":
            return models_pkg['nb']
        if choice == "Random Forest":
            return models_pkg['rf']
        if choice == "SVM":
            return models_pkg['svm']

classifier = get_classifier(model_choice)

# ============= Helper functions (unchanged logic) =================
def estimate(review_text, selected_foods):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()

    review = re.sub('[^a-zA-Z]', ' ', review_text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords]
    review = ' '.join(review)

    X = tfidf.transform([review]).toarray()
    res = classifier.predict(X)

    if "not" in review:
        res[0] = abs(res[0] - 1)

    c.execute("SELECT *, oid FROM item")
    records = c.fetchall()

    for rec in records:
        rec = list(rec)
        if rec[0] in selected_foods:
            n_cust = int(rec[1]) + 1
            n_pos = int(rec[2])
            n_neg = int(rec[3])
            if res[0] == 1:
                n_pos += 1
            else:
                n_neg += 1
            pos_percent = round((n_pos / n_cust) * 100, 1)
            neg_percent = round((n_neg / n_cust) * 100, 1)
            c.execute("""UPDATE item SET 
                            no_of_customers=?, 
                            no_of_positives=?, 
                            no_of_negatives=?, 
                            pos_perc=?, 
                            neg_perc=? WHERE oid=?""",
                      (str(n_cust), str(n_pos), str(n_neg),
                       f"{pos_percent}%", f"{neg_percent}%", str(rec[-1])))
    conn.commit()
    conn.close()

def get_data():
    conn = sqlite3.connect('Restaurant_food_data.db')
    df = pd.read_sql_query("SELECT * FROM item", conn)
    conn.close()
    return df

def clear_selected(items):
    conn = sqlite3.connect('Restaurant_food_data.db')
    c = conn.cursor()
    for food in items:
        c.execute("""UPDATE item 
                     SET no_of_customers=?, no_of_positives=?, no_of_negatives=?, 
                         pos_perc=?, neg_perc=? 
                     WHERE item_name=?""",
                  ("0", "0", "0", "0.0%", "0.0%", food))
    conn.commit()
    conn.close()

def clear_all():
    clear_selected(FOODS)

# ============= Streamlit UI (unchanged) ============================
st.title("🍴 Restaurant Review Analysis System")
st.write("An ML-powered system to analyze restaurant reviews with **Naive Bayes / Random Forest / SVM**.")

menu = st.sidebar.radio("Navigation", ["Home", "Customer", "Owner", "Analytics"])

# ... (copy the rest of the UI from your original app for Customer/Owner/Analytics)
# For brevity here, keep the same UI code from your original script.
# The key change is that model loading is done above and estimate() uses classifier variable.
