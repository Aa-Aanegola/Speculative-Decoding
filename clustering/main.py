from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import json
import joblib

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
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def cluster_embeddings(embeddings, n_clusters, save=False, save_path=None):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided with save=True dummy")
        joblib.dump(kmeans, save_path)
    
    return kmeans


if __name__ == "__main__":
    data = load_data('/insomnia001/depts/edu/COMSE6998/aa5506/Speculative-Decoding/benchmarks/data/question.jsonl')
    
    sentences = [''.join(item['turns']) for item in data]    
    embeddings = encode_sentences(sentences)
    
    n_clusters = 5 
    kmeans = cluster_embeddings(embeddings, n_clusters, save=True, save_path='kmeans_model.pkl')
    
    print("Cluster centers:")
    print(kmeans.cluster_centers_)