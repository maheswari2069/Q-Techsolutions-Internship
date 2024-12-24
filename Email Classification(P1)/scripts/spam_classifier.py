import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Paths
DATA_PATH = r'data'  # Updated for your directory structure
MODEL_PATH = r'models/spam_model.pkl'
VECTORIZER_PATH = r'models/tfidf_vectorizer.pkl'

# Function to load dataset
def load_dataset(path):
    emails = []
    labels = []
    # Iterate over the folder names and corresponding labels
    for label, folder in [('ham', 'easy_ham'), ('ham', 'hard_ham'), ('spam', 'spam_2')]:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
        
        # Skip system directories
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    # Read the email content
                    with open(file_path, 'r', encoding='latin1') as f:
                        email_content = f.read()
                        if email_content.strip():  # Skip empty files
                            emails.append(email_content)
                            labels.append(1 if label == 'spam' else 0)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    if not emails:
        raise ValueError("No valid emails found in the dataset!")
    return pd.DataFrame({'text': emails, 'label': labels})

# Load and preprocess data
print("Loading dataset...")
data = load_dataset(DATA_PATH)
print(f"Loaded {len(data)} emails.")

# Split data into training and testing sets
X = data['text']
y = data['label']
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the data using TF-IDF vectorizer
print("Vectorizing text data...")
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# Train the model
print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Ensure the models directory exists
print("Saving model and vectorizer...")
with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(model, model_file)
with open(VECTORIZER_PATH, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print("Model and vectorizer saved successfully.")
