import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from PIL import Image

# Paths for your saved model and vectorizer
MODEL_PATH = r'E:\QTech Solutions\Email Classification(P1)\models\spam_model.pkl'
VECTORIZER_PATH = r'E:\QTech Solutions\Email Classification(P1)\models\tfidf_vectorizer.pkl'

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Navigation
page = st.sidebar.radio("Select a Page", ("Home", "Accuracy Metrics"))

# Load Model and Vectorizer
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    model_status = True
except FileNotFoundError as e:
    model_status = False

# Home Page
if page == "Home":
    # App Header
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 48px;
            font-weight: bold;
            color: #3E4A89;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            color: #5A5C71;
            text-align: center;
            margin-bottom: 30px;
        }
        </style>
        <div class="main-title">Spam Email Detection System 📧</div>
        <div class="subtitle">Empowering users with AI-driven email classification</div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3416/3416048.png", width=150)
        st.markdown(
            """
            ### How it Works:
            - Paste the content of your email below.
            - Our advanced AI model classifies it as **Spam** or **Not Spam**.
            
            ### Features:
            - **Fast and Accurate**
            - **Interactive Results**
            - **Secure and Reliable**
            """
        )

    # Check Model Status
    if not model_status:
        st.error("Error: Unable to load the model and vectorizer. Please check the file paths.")
        st.stop()

    # Input Section for Email Classification
    st.markdown("### Email Content:")
    email_text = st.text_area(
        label="Paste your email content here:",
        placeholder="Type or paste the email content...",
        height=250,
    )

    # Classification Button
    if st.button("Classify Email 🚀"):
        if email_text.strip():
            # Transform and Predict
            input_vector = vectorizer.transform([email_text])
            prediction = model.predict(input_vector)[0]

            # Display Results
            st.markdown("---")
            if prediction == 1:
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: red;">⚠️ Spam Detected!</h2>
                        <p>This email has been classified as <strong>Spam</strong>. Be cautious before opening it!</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image("https://cdn-icons-png.flaticon.com/512/1040/1040266.png", width=150)
            else:
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: green;">✅ Not Spam</h2>
                        <p>This email is classified as <strong>Not Spam</strong>. It appears safe!</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=150)
        else:
            st.warning("Please enter email content before classification.")

# Accuracy Metrics Page
if page == "Accuracy Metrics":
    

    # Add a colorful header with custom styling
    st.markdown(
        """
        <style>
        .header-title {
            font-size: 40px;
            font-weight: bold;
            color: #2F8C5D;
            text-align: center;
            margin-bottom: 20px;
        }
        .metrics-box {
            background-color: #F0F8FF;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .metric-text {
            font-size: 22px;
            font-weight: bold;
            color: #4B0082;
        }
        .report-box {
            background-color: #fff;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        <div class="header-title">📊 Model Accuracy and Metrics</div>
        """,
        unsafe_allow_html=True,
    )

    # Load Dataset (for evaluation)
    import os
    import pandas as pd

    # Function to load dataset
    def load_dataset(path):
        emails = []
        labels = []
        for label, folder in [('ham', 'easy_ham'), ('ham', 'hard_ham'), ('spam', 'spam_2')]:
            folder_path = os.path.join(path, folder)
            if not os.path.exists(folder_path):
                continue

            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='latin1') as f:
                            email_content = f.read()
                            if email_content.strip():
                                emails.append(email_content)
                                labels.append(1 if label == 'spam' else 0)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        return pd.DataFrame({'text': emails, 'label': labels})

    # Load and preprocess data
    DATA_PATH = r'data'  # Your dataset path
    data = load_dataset(DATA_PATH)
    X = data['text']
    y = data['label']
    X_vectorized = vectorizer.transform(X)

    # Predict using the model
    y_pred = model.predict(X_vectorized)

    # Accuracy and Classification Report
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    # Display Accuracy in a colorful box
    st.markdown(f"""
        <div class="metrics-box">
            <p class="metric-text">Overall Accuracy: {accuracy * 100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Display Classification Report in a neat, tabulated format
    report_df = pd.DataFrame(report).transpose()
    st.markdown("### Classification Report")
    st.dataframe(report_df.style.set_table_styles(
        [{
            'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]
        }, {
            'selector': 'tbody td', 'props': [('background-color', '#f9f9f9')]
        }]
    ).highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'))
