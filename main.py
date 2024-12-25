import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('data/amazon.csv')

# Prepare the recommendation system
data['combined_features'] = data['product_name'] + ' ' + data['category'] + ' ' + data['about_product']
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(data['combined_features'])

# Streamlit app
st.set_page_config(page_title="E-Commerce Recommendation", page_icon="🛍", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #000000;  /* Custom header color set to black */
        text-align: center;
        font-size: 3.5rem;
        background: linear-gradient(to right, #8b0000, #000000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #ff4500;
        font-weight: bold;
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 30px;
    }
    .product {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
        border: 2px solid #2e86c1;
    }
    .product img {
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .recommend-section {
        background-color: #eaf2f8;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .button {
        background-color: #1e90ff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
    }
    .button:hover {
        background-color: #ff4500;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🛍 E-Commerce Product Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find the perfect product with ease and style!</div>", unsafe_allow_html=True)

# Search input
search_query = st.text_input(
    "🔍 Search for a product:", placeholder="Type a product keyword...", help="Enter a keyword to find related products."
)

if search_query:
    st.markdown("<div class='recommend-section'><h2 style='color: #2e86c1;'>Recommended Products</h2></div>", unsafe_allow_html=True)
    query_vector = vectorizer.transform([search_query])
    similarities = cosine_similarity(query_vector, features).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # Get top 5 recommendations

    for idx in top_indices:
        product = data.iloc[idx]
        img_link = product['img_link'] if pd.notna(product['img_link']) else "https://via.placeholder.com/150"  # Placeholder for missing images
        st.markdown(
            f"""
            <div class='product'>
                <img src='{img_link}' width='200'>
                <h3 style='color: #2e86c1;'>💎 {product['product_name']}</h3>
                <p><strong>🌐 Category:</strong> {product['category']}</p>
                <p><strong>⭐ Rating:</strong> {product['rating']}</p>
                <a href='{product['product_link']}' target='_blank' class='button'>View Product</a>
            </div>
            """,
            unsafe_allow_html=True
        )


