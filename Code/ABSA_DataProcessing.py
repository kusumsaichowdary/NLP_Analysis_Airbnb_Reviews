# ABSA_DataProcessing.py

# Import standard libraries
import pandas as pd
import numpy as np
import nltk
import gensim
import warnings
import random
import os  # To handle directory operations
import json  # For loading manual cluster labels if using external file

# Define a fixed seed for reproducibility
SEED = 42

# Set seeds for random, numpy, and gensim
random.seed(SEED)
np.random.seed(SEED)

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Import libraries for word embeddings and clustering
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Import NLTK modules
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Step 1: Loading Data
print("\nStep 1: Loading Data")
print("Loading the preprocessed data from 'processed_reviews_SentimentLabels.csv'...")
data = pd.read_csv('processed_reviews_SentimentLabels.csv')

# Check for necessary columns
required_columns = ['listing_id', 'review_posted_date', 'cleaned_review', 'sentiment_label']
if all(column in data.columns for column in required_columns):
    print("All required columns are present.")
else:
    missing_cols = [col for col in required_columns if col not in data.columns]
    raise ValueError(f"Missing columns: {missing_cols}")

# Display sample data
print("\nSample data:")
print(data.head())

# Step 2: Tokenization and Lemmatization
print("\nStep 2: Tokenization and Lemmatization")
print("Initializing lemmatizer and tokenizing reviews...")

lemmatizer = WordNetLemmatizer()


def tokenize_and_lemmatize(text):
    try:
        tokens = word_tokenize(text.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        return lemmatized_tokens
    except Exception as e:
        print(f"Error in tokenization and lemmatization: {e}")
        return []


data['tokens'] = data['cleaned_review'].apply(tokenize_and_lemmatize)

# Display sample tokens
print("\nSample tokens after lemmatization:")
print(data[['listing_id', 'review_posted_date', 'cleaned_review', 'tokens']].head())

# Step 3: Training the Word2Vec Model
print("\nStep 3: Training the Word2Vec Model")
print("Training Word2Vec model on tokenized reviews...")
sentences = data['tokens'].tolist()

w2v_model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=1,  # Set workers=1 for determinism
    epochs=10,
    seed=SEED  # Set the seed here
)

print("Word2Vec model training completed.")
print(f"Vocabulary size: {len(w2v_model.wv)} words")

# Step 4: Extracting Nouns from Vocabulary
print("\nStep 4: Extracting Nouns from Vocabulary")
print("Performing POS tagging to extract nouns from the vocabulary...")
vocab = sorted(list(w2v_model.wv.index_to_key))  # Sort the vocabulary for consistency
pos_tags = nltk.pos_tag(vocab)
nouns = [word for word, pos in pos_tags if pos.startswith('NN')]

print(f"Total words in vocabulary: {len(vocab)}")
print(f"Total nouns extracted: {len(nouns)}")
print("\nSample nouns:")
print(nouns[:20])

# Step 5: Obtaining Embeddings for Nouns
print("\nStep 5: Obtaining Embeddings for Nouns")
print("Retrieving embeddings for the extracted nouns...")
noun_embeddings = []
nouns_filtered = []
for noun in nouns:
    if noun in w2v_model.wv:
        noun_embeddings.append(w2v_model.wv[noun])
        nouns_filtered.append(noun)

noun_embeddings = np.array(noun_embeddings)
nouns = sorted(nouns_filtered)  # Sort nouns to ensure consistent ordering

print(f"Total nouns with embeddings: {len(noun_embeddings)}")

# Step 6: Clustering Noun Embeddings
print("\nStep 6: Clustering Noun Embeddings")
print("Clustering noun embeddings using KMeans to identify aspects...")
num_clusters = 15
kmeans = KMeans(n_clusters=num_clusters, random_state=SEED, n_init=10)  # Set n_init explicitly
nouns_clusters = pd.DataFrame({'noun': nouns})
nouns_clusters['cluster'] = kmeans.fit_predict(noun_embeddings)

print("\nNumber of nouns in each cluster:")
print(nouns_clusters['cluster'].value_counts().sort_index())

# Step 7: Assigning Aspect Names to Clusters
print("\nStep 7: Assigning Aspect Names to Clusters")
print("Printing nouns in each cluster for manual aspect mapping...")

# Create a directory to save cluster contents (optional)
clusters_dir = 'clusters_output'
os.makedirs(clusters_dir, exist_ok=True)

# Iterate through each cluster and print/save its nouns
for cluster_num in range(num_clusters):
    cluster_nouns = nouns_clusters[nouns_clusters['cluster'] == cluster_num]['noun'].tolist()
    print(f"\n--- Cluster {cluster_num} ---")
    print(cluster_nouns)

    # Save to a text file for easier inspection
    with open(os.path.join(clusters_dir, f'cluster_{cluster_num}.txt'), 'w') as f:
        for noun in cluster_nouns:
            f.write(f"{noun}\n")

print("\nAll clusters have been printed and saved to the 'clusters_output' directory.")
print("Please review the clusters and manually assign aspect names accordingly.")

# Define the manual cluster labels based on your review
manual_cluster_labels = {
    0: 'Host Interaction and Personal Experience',
    1: 'Nearby Amenities and Food Options',
    2: 'Accommodation Facilities and Aesthetics',
    3: 'Communication and Check-in Process',
    4: 'People and Personal Interactions',
    5: 'Amenities and Supplies',
    6: 'Nearby Amenities and Food Options',
    7: 'Accommodation Facilities and Aesthetics',
    8: 'Issues and Complaints',
    9: 'Nearby Amenities and Food Options',
    10: 'Location and Transportation',
    11: 'Value and Pricing',
    12: 'Location and Transportation',
    13: 'Transportation',
    14: 'Accommodation Comfort and Issues'
}

# Assign aspects based on manual mapping
nouns_clusters['aspect'] = nouns_clusters['cluster'].map(manual_cluster_labels)

# Handle any missing aspects
nouns_clusters['aspect'].fillna('Other', inplace=True)

print("\nSample nouns with their assigned aspects:")
print(nouns_clusters.head(20))

# Step 8: Associating Aspects with Reviews
print("\nStep 8: Associating Aspects with Reviews")
print("Creating a mapping of nouns to aspects and assigning aspects to each review...")

# Create a dictionary mapping nouns to their aspects
noun_to_aspect = pd.Series(nouns_clusters['aspect'].values, index=nouns_clusters['noun']).to_dict()


def get_aspects_from_tokens(tokens):
    aspects = set()
    for token in tokens:
        if token in noun_to_aspect:
            aspects.add(noun_to_aspect[token])
    return list(aspects)


data['aspects'] = data['tokens'].apply(get_aspects_from_tokens)

print("\nSample reviews with their associated aspects:")
print(data[['listing_id', 'review_posted_date', 'cleaned_review', 'tokens', 'aspects']].head())

# Step 9: Creating a DataFrame for Aspect-Level Sentiments
print("\nStep 9: Creating a DataFrame for Aspect-Level Sentiments")
print("Exploding aspects to create a row for each aspect mentioned in a review...")

# Include 'listing_id' and 'review_posted_date' in the exploded DataFrame
data_exploded = data.explode('aspects')
data_exploded = data_exploded.dropna(subset=['aspects'])

# Select the desired columns, including 'listing_id' and 'review_posted_date'
aspect_sentiments = data_exploded[['listing_id', 'review_posted_date', 'cleaned_review', 'aspects', 'sentiment_label']]

print("\nSample aspect-level sentiment data:")
print(aspect_sentiments.head())

# Step 10: Saving the Aspect-Level Sentiment Data
print("\nStep 10: Saving the Aspect-Level Sentiment Data")
print("Saving the aspect-level sentiment data to 'AspectbasedSentimentAnalysis.csv'...")
aspect_sentiments.to_csv('AspectbasedSentimentAnalysis.csv', index=False)
print("Data saved successfully.")
