import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Settings
DATA_FILE = "preprocessed_data.pkl"
MODEL_NAME = "all-mpnet-base-v2"
TOP_K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load preprocessed data
def load_data():
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data["chunks"])
    embeddings = torch.tensor(data["embeddings"]).to(DEVICE)
    return df, embeddings

# Fake queries using the first few words from random chunks
def create_fake_queries(df, n=10):
    queries, relevant = [], []
    for _ in range(n):
        row = df.sample(1).iloc[0]
        query = " ".join(row["sentence_chunk"].split()[:5])
        rel = df[df["file_name"] == row["file_name"]].index.tolist()
        queries.append(query)
        relevant.append(rel)
    return queries, relevant

# Semantic search (cosine similarity)
def semantic_search(query_vec, df, embeddings, threshold):
    sims = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), embeddings)
    mask = sims >= threshold
    filtered = df[mask.cpu().numpy()]
    scores = sims[mask]
    top_idx = scores.argsort(descending=True)[:TOP_K].cpu().numpy()
    return filtered.iloc[top_idx], scores[top_idx].cpu().numpy()

# BM25 keyword search
def bm25_search(query, df, k1, b):
    corpus = [d.split() for d in df["sentence_chunk"]]
    bm25 = BM25Okapi(corpus, k1=k1, b=b)
    return bm25.get_scores(query.split())

# Combine both search scores
def combined_search(query, query_vec, df, embeddings, k1, b, threshold, weight):
    sem_results, sem_scores = semantic_search(query_vec, df, embeddings, threshold)
    if sem_results.empty:
        return pd.DataFrame()
    bm_scores = bm25_search(query, sem_results, k1, b)
    final = weight * sem_scores + (1 - weight) * bm_scores
    return sem_results.iloc[final.argsort()[::-1][:TOP_K]]

# Basic metrics
def eval_query(rel, ret):
    prec = len(set(rel) & set(ret[:TOP_K])) / TOP_K
    rec = len(set(rel) & set(ret[:TOP_K])) / len(rel)
    mrr = next((1/(i+1) for i, doc in enumerate(ret) if doc in rel), 0)
    dcg = sum((1/np.log2(i+2) if doc in rel else 0) for i, doc in enumerate(ret[:TOP_K]))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(rel), TOP_K)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return prec, rec, mrr, ndcg

# Evaluate all queries
def evaluate(queries, rels, df, emb, model, k1, b, threshold, weight):
    total = np.zeros(4)
    for q, r in zip(queries, rels):
        q_vec = model.encode(q, convert_to_tensor=True, device=DEVICE)
        results = combined_search(q, q_vec, df, emb, k1, b, threshold, weight)
        if results.empty: continue
        scores = eval_query(r, results.index.tolist())
        total += scores
    return -np.mean(total / len(queries))  # We minimize, so flip it

# Search space
space = [
    Real(0.5, 2.0, name="k1"),
    Real(0.1, 1.0, name="b"),
    Real(0.1, 0.9, name="threshold"),
    Real(0.1, 0.9, name="weight")
]

@use_named_args(space)
def objective(k1, b, threshold, weight):
    return evaluate(queries, relevant_docs, df, embeddings, model, k1, b, threshold, weight)

# Main logic
if __name__ == "__main__":
    print("📦 Loading everything...")
    df, embeddings = load_data()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    queries, relevant_docs = create_fake_queries(df)

    print("🎯 Tuning search parameters...")
    results = gp_minimize(objective, space, n_calls=30, random_state=42, verbose=True)

    best = {
        "k1": results.x[0],
        "b": results.x[1],
        "similarity_threshold": results.x[2],
        "weight_semantic": results.x[3]
    }

    print("\n✅ Done! Best parameters found:")
    for k, v in best.items():
        print(f"{k}: {v:.4f}")

    # Save to file
    with open("best_retrieval_params.pkl", "wb") as f:
        pickle.dump(best, f)

    # Save basic plot
    plt.plot(results.func_vals)
    plt.title("Tuning Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.savefig(os.path.join(PLOT_DIR, "retrieval_tuning_plot.png"))
    plt.close()