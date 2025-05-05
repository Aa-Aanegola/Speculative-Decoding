from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import json
import joblib

BATCH_SIZE = 32

def load_data(file_path):
    """
    Load data from a JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

model = SentenceTransformer('all-MiniLM-L6-v2')
def encode_sentences(sentences):
    embeddings = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
    
def cluster_embeddings(embeddings, n_clusters, save=False, save_path=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided with save=True dummy")
        joblib.dump(kmeans, save_path)
    
    return kmeans

def add_embeddings_to_dset(data, kmeans):
    batch_size = 32
    
    embeddings = encode_sentences([''.join(item['turns']) for item in data])
    labels = kmeans.predict(embeddings)
    
    for i, item in enumerate(data):
        item['cluster'] = int(labels[i])
        item['embedding'] = embeddings[i].tolist()
    return data

if __name__ == "__main__":
    data = load_data('/insomnia001/depts/edu/COMSE6998/aa5506/Speculative-Decoding/benchmarks/data/question.jsonl')
    
    sentences = [''.join(item['turns']) for item in data]    
    embeddings = encode_sentences(sentences)
    
    n_clusters = 5
    kmeans = cluster_embeddings(embeddings, n_clusters, save=True, save_path='kmeans_model.job')
    
    print("Cluster centers:")
    print(kmeans.cluster_centers_)
    
    data_with_clusters = add_embeddings_to_dset(data, kmeans)
    with open('data_with_clusters.jsonl', 'w') as f:
        for item in data_with_clusters:
            f.write(json.dumps(item) + '\n')