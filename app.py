# app.py  (full version with UI)
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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

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
USE_PRETRAINED = art.get('loaded', False)

# ========== Database init ==========
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

# ========== If pickles are NOT present, train on the fly ==========
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

    return {'nb': nb, 'rf': rf, 'svm': svm, 'X_test': X_test, 'y_test': y_test}

if not USE_PRETRAINED:
    models_pkg = train_models_onfly()
else:
    models_pkg = None

# --------- Model selector ----------
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
        return models_pkg[{'Naive Bayes': 'nb', 'Random Forest': 'rf', 'SVM': 'svm'}[choice]]

classifier = get_classifier(model_choice)

# ============= Helper functions =================
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

# ============= Streamlit UI ============================
st.title("🍴 Restaurant Review Analysis System")
st.write("An ML-powered system to analyze restaurant reviews with **Naive Bayes / Random Forest / SVM**.")

menu = st.sidebar.radio("Navigation", ["Home", "Customer", "Owner", "Analytics"])

if menu == "Home":
    st.subheader("Welcome")
    st.write("Choose whether you are a **Customer** or the **Owner** from the sidebar.")

elif menu == "Customer":
    st.subheader("Leave a Review")
    st.write("Select the food items you had:")

    if 'selected_foods' not in st.session_state:
        st.session_state.selected_foods = {food: False for food in FOODS}

    selected_foods = []
    for food in FOODS:
        st.session_state.selected_foods[food] = st.checkbox(food, value=st.session_state.selected_foods[food])
        if st.session_state.selected_foods[food]:
            selected_foods.append(food)

    review_text = st.text_area("Write your review:")

    if st.button("Submit Review"):
        if review_text and selected_foods:
            estimate(review_text, selected_foods)
            st.success("✅ Review submitted successfully!")
            for food in FOODS:
                st.session_state.selected_foods[food] = False
        else:
            st.error("⚠️ Please select food items and write a review.")

elif menu == "Owner":
    st.subheader("Owner Login")
    rras_code = "sarth"
    code_input = st.text_input("Enter Owner Code:", type="password")
    if st.button("Verify"):
        if code_input == rras_code:
            st.success("Access Granted ✅")

            st.subheader("📊 Current Database")
            df = get_data()
            st.dataframe(df)

            st.subheader("🧹 Clear Data")
            clear_option = st.radio("Choose:", ["Clear Selected", "Clear All"])
            if clear_option == "Clear Selected":
                items_to_clear = st.multiselect("Select items to clear:", FOODS)
                if st.button("Clear Selected Data"):
                    clear_selected(items_to_clear)
                    st.success("Selected items cleared.")
            elif clear_option == "Clear All":
                if st.button("Clear All Data"):
                    clear_all()
                    st.success("All items cleared.")
        else:
            st.error("❌ Incorrect Code")

elif menu == "Analytics":
    st.subheader("📊 Reviews per Item")
    df = get_data()
    review_counts = df[['item_name', 'no_of_positives', 'no_of_negatives']].copy()
    review_counts['no_of_positives'] = review_counts['no_of_positives'].astype(int)
    review_counts['no_of_negatives'] = review_counts['no_of_negatives'].astype(int)

    fig1, ax1 = plt.subplots(figsize=(12,6))
    width = 0.35
    x = np.arange(len(review_counts['item_name']))

    ax1.bar(x - width/2, review_counts['no_of_positives'], width, label='Positive Reviews', color='green')
    ax1.bar(x + width/2, review_counts['no_of_negatives'], width, label='Negative Reviews', color='red')

    ax1.set_xticks(x)
    ax1.set_xticklabels(review_counts['item_name'], rotation=45, ha='right')
    ax1.set_ylabel("Number of Reviews")
    ax1.set_title("Positive vs Negative Reviews per Item")
    ax1.legend()
    ax1.grid(axis='y')

    st.pyplot(fig1)

    st.subheader("Database Snapshot")
    st.dataframe(df)

    st.subheader("📈 Model Performance")
    st.write(f"**Selected Model:** {model_choice}")
    if not USE_PRETRAINED:
        X_test, y_test = models_pkg['X_test'], models_pkg['y_test']
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
        cm = confusion_matrix(y_test, y_pred)
    else:
        st.info("Model performance was computed offline when models were trained.")
        accuracy, report, cm = 0, "Pretrained models loaded.", np.array([[0,0],[0,0]])

    st.write(f"**Accuracy:** {accuracy*100:.2f}%")
    st.text("Classification Report:")
    st.text(report)

    st.subheader("Confusion Matrix")
    fig2, ax2 = plt.subplots(figsize=(5,4))
    im = ax2.imshow(cm, cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    classes = ['Negative', 'Positive']
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    st.pyplot(fig2)
