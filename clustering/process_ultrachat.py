from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import json
import joblib
from utils import *

if __name__ == "__main__":
    data = load_data('/insomnia001/depts/edu/COMSE6998/aa5506/Speculative-Decoding/benchmarks/data/ultrachat_30000_prompts.jsonl')
    sentences = [item['prompt'] for item in data]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = encode_sentences(model, sentences)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    joblib.dump(kmeans, 'kmeans_model_uc.joblib')
    
    bs = 32
    data_aug = []
    for i in range(0, len(sentences), bs):
        batch = sentences[i:i + bs]
        batch_embeddings = embeddings[i:i + bs]
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
    
    with open('ultrachat_30000_prompts_clustered.jsonl', 'w') as f:
        for item in data_aug:
            f.write(json.dumps(item) + '\n')
    print("Clustering complete and data saved to ultrachat_30000_prompts_clustered.jsonl")
    
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
    print("Subsampling complete and data saved to ultrachat_5000_prompts_clustered.jsonl")