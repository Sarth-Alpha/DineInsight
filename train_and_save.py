# train_and_save.py
import re
import joblib
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# --- load dataset (same file that your app uses) ---
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

ps = PorterStemmer()
stopwords = set(ENGLISH_STOP_WORDS)
if 'not' in stopwords:
    stopwords.remove('not')

corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords]
    corpus.append(' '.join(review))

tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Train three models and save them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)

print("Training SVM...")
svm = LinearSVC(random_state=0, max_iter=10000)
svm.fit(X_train, y_train)

print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)

# Save
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(rf, 'model_rf.pkl')
joblib.dump(svm, 'model_svm.pkl')
joblib.dump(nb, 'model_nb.pkl')

print("Saved tfidf.pkl and model_{rf,svm,nb}.pkl")
