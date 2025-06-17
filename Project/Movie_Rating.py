import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('D:\\23KP1A44G4\\Project\\IMDb Movies India.csv', encoding='latin1')



le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column not in ['Name']:  # Exclude the 'Name' column
        df[column] = le.fit_transform(df[column])

df['Rating'] = df['Rating'].fillna(df['Rating'].mean())



X = df.drop(['Name', 'Rating'], axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Root Mean Squared Error:", rmse)

