
# Local PDF Search (Local-RAG)

This is a simple tool to search PDF files using both AI (semantic search) and keyword matching (BM25).

## Setup

1. Install Python 3.12.1
2. Create and activate a virtual environment (optional but helpful)
3. Install requirements:
   ```bash
   pip install -r requirements.txt
````

## How to Use

1. Put your PDFs in the `files/` folder
2. Run preprocessing:

   ```bash
   python preprocess.py
   ```
3. Run the search:

   ```bash
   python retrieval.py
   ```

Type your question and get results. Type `quit` to exit.

## Output

* Creates `preprocessed_data.pkl` with all chunks + embeddings

Sure! Here's a short and simple explanation that includes the tuner:

---

## 🛠️ How It Works (Quick Overview)

1. **Preprocessing**:

   * Reads your PDFs and splits them into chunks.
   * Generates AI embeddings for each chunk (like turning text into math).
   * Saves everything to a file for fast searching.

2. **Searching**:

   * You enter a question.
   * It finds the best-matching chunks using:

     * **Semantic search** (AI understands meaning)
     * **BM25** (keyword matching)
   * Combines both results and shows the most relevant answers.

3. **Tuning (Optional)**:

   * You can run the tuners to find the best settings for chunking and search accuracy.
   * This helps improve performance and relevance on your specific data.

Everything runs locally — no internet or API needed.
