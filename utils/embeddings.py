from tqdm import tqdm

def encode_sentences(model, sentences, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings