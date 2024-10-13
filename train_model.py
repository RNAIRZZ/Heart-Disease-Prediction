import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # or RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('heart.csv')

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = DecisionTreeClassifier()  # or RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training completed and saved as heart_disease_model.pkl")
