import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_products(data, query, top_n=5):
    # Combine features for recommendation
    data['combined_features'] = data['product_name'] + ' ' + data['category'] + ' ' + data['about_product']
    
    # Vectorize features
    vectorizer = TfidfVectorizer(stop_words='english')
    features = vectorizer.fit_transform(data['combined_features'])
    
    # Compute similarity
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, features).flatten()
    
    # Get top recommendations
    top_indices = similarities.argsort()[-top_n:][::-1]
    return data.iloc[top_indices].to_dict(orient='records')
