import sys
sys.path.append('../')
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import json
import joblib
from utils import *
import yaml
import argparse

if __name__ == "__main__":
    # Load arguments 
    parser = argparse.ArgumentParser(description='Ultrachat Clustering')
    parser.add_argument('--yaml', type=str, required=True, help='yaml file to load')
    
    args = parser.parse_args()
    
    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data - specific to ultrachat format
    data = load_jsonl(config['data_path'])
    sentences = [item['prompt'] for item in data]
    
    # Load the Kmeans model and cluster, store the model too 
    model = SentenceTransformer(config['sentence_embedding_model'])
    embeddings = encode_sentences(model, sentences, config['batch_size'])
    n_clusters = config['n_clusters']
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    joblib.dump(kmeans, config['kmeans_model_path'])
    
    # augment the data with the cluster labels 
    data_aug = []
    for i in range(0, len(sentences), config['batch_size']):
        batch = sentences[i:i + config['batch_size']]
        batch_embeddings = embeddings[i:i + config['batch_size']]
        batch_labels = kmeans.predict(batch_embeddings)
        data_aug.extend([
            {
                'id': i,
                'prompt': prompt,
                'cluster': int(label),
                'dist': float(np.linalg.norm(embeddings[i] - kmeans.cluster_centers_[label]))
            }
            for prompt, label in zip(batch, batch_labels)
        ])
    
    # Save the augmented data 
    with open(config['clustered_data_path'], 'w') as f:
        for item in data_aug:
            f.write(json.dumps(item) + '\n')
    print(f"Clustering complete and data saved to {config['clustered_data_path']}")
    
    # This is if we're experimenting with smaller datasets - create two smaller versions 
    if config['subsample']:
        subsampled_data = []
        ultra_subsampled_data = []
        mp = {i : [] for i in range(n_clusters)}
        for item in data_aug:
            mp[item['cluster']].append(item)
        
        for i in range(n_clusters):
            sm = sum([1 / item['dist'] for item in mp[i]])
            scores = [1 / item['dist'] / sm for item in mp[i]]
            selected_indices = np.random.choice(len(mp[i]), size=500, p=scores)
            selected_indices_ultra = np.random.choice(len(mp[i]), size=20, p=scores)
            subsampled_data.extend([mp[i][j] for j in selected_indices])
            ultra_subsampled_data.extend([mp[i][j] for j in selected_indices_ultra])
        
        with open('ultrachat_5000_prompts_clustered.jsonl', 'w') as f:
            for item in subsampled_data:
                f.write(json.dumps(item) + '\n')
        with open('ultrachat_100_prompts_clustered.jsonl', 'w') as f:
            for item in ultra_subsampled_data:
                f.write(json.dumps(item) + '\n')