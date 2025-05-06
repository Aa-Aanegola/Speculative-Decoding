import json
from tqdm import tqdm

BATCH_SIZE = 32

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def encode_sentences(model, sentences):
    embeddings = []
    for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
        batch = sentences[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings