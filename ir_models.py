import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

# Load resources (ONLY ONCE)
df = pd.read_csv("amazon.csv")
doc_embeddings = np.load("doc_embeddings.npy")
bert = SentenceTransformer("all-MiniLM-L6-v2")


#Tokenize
def tokenize_bm25(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

#BM25 only
with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

def search_bm25(query, top_k=10):
    tokens = tokenize_bm25(query)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return df.iloc[top_idx]

#BERT + BM25

def search_bm25_bert(query, top_k_bm25=50, top_k_final=5):
     # 1. BM25 retrieve
    tokens = tokenize_bm25(query)
    scores = bm25.get_scores(tokens)
    bm25_idx = np.argsort(scores)[::-1][:top_k_bm25]

    # 2. Encode query
    query_emb = bert.encode(query, normalize_embeddings=True)

    # 3. BERT rerank
    cand_embs = doc_embeddings[bm25_idx]
    bert_scores = cosine_similarity([query_emb], cand_embs)[0]

    rerank_idx = bm25_idx[np.argsort(bert_scores)[::-1][:top_k_final]]

    return df.iloc[rerank_idx]