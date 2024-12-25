import streamlit as st
from iris_model import load_data, preprocess_data, train_and_evaluate_models
import pandas as pd
import plotly.express as px

# Load data and preprocess
df, target_names = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Streamlit Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for User Input
st.sidebar.header("🔢 User Input Features")
st.sidebar.markdown("Adjust the sliders below to predict the Iris species:")

# Collect user input
sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(df['sepal length (cm)'].min()),
    float(df['sepal length (cm)'].max()),
    step=0.1
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(df['sepal width (cm)'].min()),
    float(df['sepal width (cm)'].max()),
    step=0.1
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max()),
    step=0.1
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max()),
    step=0.1
)

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

# Scale the user input using the same scaler
user_input_scaled = scaler.transform(user_input)

# Prediction using the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
predicted_class = best_model.predict(user_input_scaled)
predicted_species = target_names[predicted_class[0]]

# Main Page Layout
st.title("🌸 Iris Flower Classification Dashboard 🌸")
st.markdown("## Predicting Iris Flower Species using Machine Learning")

# Display Model Performance
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌟 Model Performances")
    results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    fig = px.bar(
        results_df,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison",
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🪴 Prediction Result")
    st.success(f"The predicted species is **{predicted_species}**.")
    st.info(f"Best Model Used: **{best_model_name}**")

# Footer Section with Data Insights
st.markdown("---")
st.subheader("📊 Dataset Insights")
st.markdown("Here is an overview of the Iris dataset used in this application:")
st.dataframe(df.head(), use_container_width=True)

st.markdown("### 🔍 Feature Ranges")
col3, col4 = st.columns(2)

with col3:
    st.metric(label="Sepal Length (cm)", value=f"{df['sepal length (cm)'].min()} - {df['sepal length (cm)'].max()}")
    st.metric(label="Sepal Width (cm)", value=f"{df['sepal width (cm)'].min()} - {df['sepal width (cm)'].max()}")

with col4:
    st.metric(label="Petal Length (cm)", value=f"{df['petal length (cm)'].min()} - {df['petal length (cm)'].max()}")
    st.metric(label="Petal Width (cm)", value=f"{df['petal width (cm)'].min()} - {df['petal width (cm)'].max()}")
