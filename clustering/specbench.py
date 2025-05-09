from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import json
import joblib
import yaml
import argparse

BATCH_SIZE = 32

# Load data from a JSONL file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Convert sentences provided to embeddings using the sentence transformer model
def encode_sentences(model, sentences):
    embeddings = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
    
# Helper function to cluster embeddings using KMeans
def cluster_embeddings(embeddings, n_clusters, save=False, save_path=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided with save=True dummy")
        joblib.dump(kmeans, save_path)
    
    return kmeans

# Helper function to augment the dataset with cluster labels
def add_embeddings_to_dset(model, data, kmeans):
    batch_size = 32
    
    embeddings = encode_sentences(model, [''.join(item['turns']) for item in data])
    labels = kmeans.predict(embeddings)
    
    for i, item in enumerate(data):
        item['cluster'] = int(labels[i])
    return data

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Ultrachat Clustering')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    args = parser.parse_args()
    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Model and data loading
    model = SentenceTransformer(config['sentence_embedding_model'])
    data = load_data(config['data_path'])
    
    sentences = [''.join(item['turns']) for item in data]    
    embeddings = encode_sentences(model, sentences)
    
    n_clusters = config['n_clusters']
    kmeans = cluster_embeddings(embeddings, n_clusters, save=True, save_path=config['kmeans_model_path'])
    
    # Debugging, just to see the cluster centers
    print("Cluster centers:")
    print(kmeans.cluster_centers_)
    
    # Augment the data with cluster labels and save
    data_with_clusters = add_embeddings_to_dset(model, data, kmeans)
    with open(config['output_path'], 'w') as f:
        for item in data_with_clusters:
            f.write(json.dumps(item) + '\n')