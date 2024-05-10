import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Read movie data from CSV file
movie_df = pd.read_csv('movie_data.csv')

# Sample user preferences
user_preferences = {
    'genre': ['Drama', 'Crime'],
    'keywords': ['prison', 'mafia']
}

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform movie descriptions
tfidf_matrix = vectorizer.fit_transform(movie_df['description'])

# Calculate sigmoid kernel (similarity)
sigmoid_sim = sigmoid_kernel(tfidf_matrix, tfidf_matrix)

# Calculate user profile based on preferences
user_profile = sigmoid_sim[:, movie_df['genre'].apply(lambda x: any(pref in x for pref in user_preferences['genre']))]
user_profile += sigmoid_sim[:, movie_df['description'].apply(lambda x: any(pref in x for pref in user_preferences['keywords']))]

# Get top recommended movie
top_movie_idx = user_profile.sum(axis=1).argmax()
top_movie_title = movie_df.loc[top_movie_idx, 'title']

print("Top recommended movie:", top_movie_title)
