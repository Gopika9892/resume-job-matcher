
#loading dataset

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('UpdatedResumeDataSet.csv')  # columns: Resume, Category
X_train, X_test, y_train, y_test = train_test_split(
    df['Resume'], df['Category'], test_size=0.2, random_state=42)

#data preproccessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

#training
pipeline.fit(X_train, y_train)
print("Training accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))

#dumping into pipeline
joblib.dump(pipeline, 'resume_job_matcher.joblib')

def predict_role(resume_text: str) -> str:
    return pipeline.predict([resume_text])[0]

#sample input
sample = "Experienced in Python, SQL, AWS, data analysis"
print("Predicted job:", predict_role(sample))

#testing
from sklearn.metrics import classification_report
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

#confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=pipeline.classes_,
            yticklabels=pipeline.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.tight_layout()
plt.show()