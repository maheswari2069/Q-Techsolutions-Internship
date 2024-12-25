import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Preprocess the data
def preprocess_data(df):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Train models and evaluate
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5)
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = accuracy
    return models, results
