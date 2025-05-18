import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("winequality-red.csv", sep=";", quotechar='"')

# Convert quality to category
df['quality'] = pd.cut(df['quality'], bins=[0, 4, 6, 10], labels=['low', 'medium', 'high'])

X = df.drop('quality', axis=1)
y = df['quality']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'app/model.pkl')
