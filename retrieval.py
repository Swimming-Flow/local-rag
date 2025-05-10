# Simple PDF Search Tool - Home Project Version
# Author: [Your Name], 2025
# Description: Load preprocessed chunks and run semantic + keyword search on them.

import pandas as pd
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import textwrap

# Basic settings
MODEL_NAME = "all-mpnet-base-v2"
DATA_FILE = "preprocessed_data.pkl"
TOP_K = 10
BATCH_SIZE = 32

# BM25 and search weights
K1 = 1.5
B = 0.75
THRESHOLD = 0.1
WEIGHT_SEMANTIC = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data from file
def load_data():
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data["chunks"])
    embeddings = torch.tensor(data["embeddings"]).to(DEVICE)
    return df, embeddings

# Embed the query text
def embed_query(query, model):
    return model.encode([query], convert_to_tensor=True, device=DEVICE)[0]

# Simple cosine similarity based semantic search
def semantic_search(query_vec, df, all_embeddings, threshold=THRESHOLD):
    sims = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), all_embeddings)
    mask = sims >= threshold
    filtered = df[mask.cpu().numpy()]
    filtered_sims = sims[mask]
    
    top_indices = filtered_sims.argsort(descending=True)[:TOP_K].cpu().numpy()
    return filtered.iloc[top_indices], filtered_sims[top_indices].cpu().numpy()

# BM25 keyword search
def bm25_search(query, df):
    corpus = df["chunk"].tolist()
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized, k1=K1, b=B)
    return bm25.get_scores(query.split())

# Combine semantic and BM25 scores
def combined_search(query, query_vec, df, all_embeddings):
    sem_df, sem_scores = semantic_search(query_vec, df, all_embeddings)
    bm25_scores = bm25_search(query, sem_df)
    
    final_scores = WEIGHT_SEMANTIC * sem_scores + (1 - WEIGHT_SEMANTIC) * bm25_scores
    sorted_idx = final_scores.argsort()[::-1][:TOP_K]
    return sem_df.iloc[sorted_idx]

# Highlight terms in results
def show_result(row, query):
    chunk = row["chunk"]
    for word in query.lower().split():
        chunk = chunk.replace(word, f"\033[1m{word}\033[0m")
    
    print(f"\nFile: {row['file']}, Page: {row['page']}")
    print(textwrap.fill(chunk, width=100))
    print("=" * 100)

# Main loop
def main():
    print("Loading model and data...")
    df, embeddings = load_data()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    while True:
        query = input("\nAsk something (or type 'quit'): ")
        if query.strip().lower() == "quit":
            break

        query_vec = embed_query(query, model)
        results = combined_search(query, query_vec, df, embeddings)

        for _, row in results.iterrows():
            show_result(row, query)

if __name__ == "__main__":
    main()
