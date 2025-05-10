import os
import time
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from preprocess import process_pdf_files

# Where your PDFs live
PDF_DIR = "./files"
MODEL_NAME = "all-mpnet-base-v2"
MAX_TRIES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up sentence splitter and embedding model
nlp = English()
nlp.add_pipe("sentencizer")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# Try different settings for these:
space = [
    Integer(50, 200, name="MIN_TOKEN_LENGTH"),
    Integer(1, 5, name="SENTENCE_OVERLAP"),
    Integer(16, 64, name="BATCH_SIZE")
]

@use_named_args(space)
def score_params(MIN_TOKEN_LENGTH, SENTENCE_OVERLAP, BATCH_SIZE):
    start = time.time()

    chunks = process_pdf_files(PDF_DIR, nlp, MIN_TOKEN_LENGTH, SENTENCE_OVERLAP)
    df = pd.DataFrame(chunks)

    texts = df["sentence_chunk"].tolist()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=DEVICE)

    # A few basic metrics
    time_taken = time.time() - start
    avg_len = df["chunk_token_count"].mean()
    std_len = df["chunk_token_count"].std()

    # Cosine similarity between adjacent chunks
    sims = [
        torch.cosine_similarity(embeddings[i], embeddings[i+1], dim=0).item()
        for i in range(len(embeddings) - 1)
    ]
    avg_sim = np.mean(sims)

    # Lower score is better
    score = (
        time_taken / 100 +
        abs(avg_len - 150) / 10 +
        std_len / 50 +
        (1 - avg_sim) * 2
    )

    print(f"→ Tried: token={MIN_TOKEN_LENGTH}, overlap={SENTENCE_OVERLAP}, batch={BATCH_SIZE} → Score: {score:.4f}")
    return score

def main():
    if not os.path.exists(PDF_DIR):
        print(f"PDF folder not found: {PDF_DIR}")
        return

    print(f"🔍 Tuning started (using {DEVICE})...")

    result = gp_minimize(score_params, space, n_calls=MAX_TRIES, random_state=42)

    best = {
        "MIN_TOKEN_LENGTH": result.x[0],
        "SENTENCE_OVERLAP": result.x[1],
        "BATCH_SIZE": result.x[2]
    }

    print("\n✅ Done! Best settings found:")
    print(best)

    # Save best settings to a file
    with open("best_preprocess_params.pkl", "wb") as f:
        pickle.dump(best, f)

    # Save a basic convergence plot
    plt.plot(result.func_vals)
    plt.title("Tuning Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.savefig("preprocess_tuning_plot.png")
    plt.close()

if __name__ == "__main__":
    main()
