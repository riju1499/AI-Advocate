import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
data_path = os.path.join("data", "arena_dataset.csv")
df = pd.read_csv(data_path)

X = df["text"]
y = df["arena"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build pipeline: TF-IDF + Logistic Regression
clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=5000,       # cap vocabulary size
    )),
    ("logreg", LogisticRegression(
        max_iter=1000

    )),
])

# 4. Train
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))

# 6. Save model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "arena_classifier.joblib")
joblib.dump(clf, model_path)
print(f"Saved classifier to {model_path}")