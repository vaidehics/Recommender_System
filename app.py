from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load MovieLens dataset
movies = pd.read_csv(r"C:\Users\Admin\Downloads\ml-latest-small\ml-latest-small\movies.csv")  # Movie metadata
ratings = pd.read_csv(r"C:\Users\Admin\Downloads\ml-latest-small\ml-latest-small\ratings.csv")  # User ratings

# Feature Engineering: Extract Temporal Features from Ratings
ratings = ratings.copy()
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['year'] = ratings['timestamp'].dt.year
ratings['month'] = ratings['timestamp'].dt.month
ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek  # Ensure 'day_of_week' exists

# Collaborative Filtering using Truncated SVD
ratings_pivot = ratings.pivot(index='userId', columns='movieId', values='rating')
ratings_pivot = ratings_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)  # Fill NaN with user mean rating

svd = TruncatedSVD(n_components=50)
user_factors = svd.fit_transform(ratings_pivot)
movie_factors = svd.components_.T
predicted_ratings = np.dot(user_factors, movie_factors.T)

def recommend_collaborative(user_id, num_recs=5):
    if user_id - 1 not in range(predicted_ratings.shape[0]):
        return ["User not found in database."]
    user_idx = user_id - 1
    movie_scores = predicted_ratings[user_idx]
    top_movie_indices = np.argsort(movie_scores)[-num_recs:][::-1]
    top_movie_ids = ratings_pivot.columns[top_movie_indices]
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()

# Content-Based Filtering using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
movies['description'] = movies['title'] + " " + movies['genres']
tfidf_matrix = tfidf.fit_transform(movies['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_content(movie_title, num_recs=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found in database."]
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num_recs+1]]
    return movies.iloc[movie_indices]['title'].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form.get("user_id")
    movie_title = request.form.get("movie_title")
    
    if user_id:
        recommendations = recommend_collaborative(int(user_id))
    elif movie_title:
        recommendations = recommend_content(movie_title)
    else:
        recommendations = ["Please enter a User ID or Movie Title"]
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)