import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import re
import torch
import pickle
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

# Folder where my PDFs are stored
PDF_DIR = "./files"
MIN_TOKEN_LENGTH = 160  # How long each text chunk should be (roughly)
SENTENCE_OVERLAP = 5    # How many sentences to overlap between chunks
BATCH_SIZE = 16
MODEL_NAME = "all-mpnet-base-v2"

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quick cleanup function
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Read a PDF and return a list of its pages as dicts
def read_pdf(filepath):
    pages = []
    with fitz.open(filepath) as doc:
        for page_num, page in enumerate(doc):
            pages.append({
                "file_name": os.path.basename(filepath),
                "doc_page": f"{os.path.basename(filepath)}_{page_num + 1}",
                "text": clean_text(page.get_text())
            })
    return pages

# Break text into sentences using spaCy
def extract_sentences(pages, nlp):
    for page in pages:
        doc = nlp(page["text"])
        page["sentences"] = [str(sent) for sent in doc.sents]
    return pages

# Chunk sentences into groups with some overlap
def make_chunks(pages, nlp):
    chunks = []
    for page in pages:
        sents = page["sentences"]
        temp_chunk = []
        token_count = 0

        for i, sent in enumerate(sents):
            sent_len = len(nlp(sent))
            if token_count + sent_len > MIN_TOKEN_LENGTH and temp_chunk:
                chunk_text = " ".join(temp_chunk)
                chunks.append({
                    "file": page["file_name"],
                    "page": page["doc_page"],
                    "chunk": chunk_text
                })
                # Overlap a few sentences
                temp_chunk = temp_chunk[-SENTENCE_OVERLAP:]
                token_count = sum(len(nlp(s)) for s in temp_chunk)

            temp_chunk.append(sent)
            token_count += sent_len

        if temp_chunk:
            chunk_text = " ".join(temp_chunk)
            chunks.append({
                "file": page["file_name"],
                "page": page["doc_page"],
                "chunk": chunk_text
            })
    return chunks

# Main function
def main():
    if not os.path.exists(PDF_DIR):
        print(f"Directory {PDF_DIR} not found. Put your PDFs there.")
        return

    # Set up the basic sentence splitter
    nlp = English()
    nlp.add_pipe("sentencizer")

    all_pages = []
    for file in os.listdir(PDF_DIR):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(PDF_DIR, file)
            all_pages.extend(read_pdf(full_path))

    pages_with_sentences = extract_sentences(all_pages, nlp)
    chunked_data = make_chunks(pages_with_sentences, nlp)

    # Turn into DataFrame for easy handling
    df = pd.DataFrame(chunked_data)

    # Load sentence transformer model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = model.encode(df["chunk"].tolist(), batch_size=BATCH_SIZE, convert_to_tensor=True)

    # Save everything to a file
    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump({
            "chunks": df.to_dict(orient="records"),
            "embeddings": embeddings.cpu().numpy()
        }, f)

    print("Done! Data saved to 'preprocessed_data.pkl'.")

if __name__ == "__main__":
    main()
