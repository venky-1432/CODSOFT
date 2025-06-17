import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('D:\\23KP1A44G4\\Project\\IRIS.csv')
## Step 2: Data Preprocessing
# Convert categorical variable into numeric
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
## Step 3: Split Data into Features and Target
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Step 4: Train the Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
## Step 5: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

