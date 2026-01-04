import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

# Load resources (ONLY ONCE)

with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)
    
df = pd.read_csv("amazon.csv")
bert = SentenceTransformer("all-MiniLM-L6-v2")
bert_ft = SentenceTransformer("./bert_finetuned")
doc_embeddings = np.load("doc_embeddings.npy")
doc_embeddings_ft = np.load("doc_embeddings_ft.npy")

#Tokenize
def tokenize_bm25(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def rerank_with_bert(
    query,
    bert_model,
    doc_embeddings,
    top_k_bm25=50,
    top_k_final=5
):
    tokens = tokenize_bm25(query)
    scores = bm25.get_scores(tokens)
    bm25_idx = np.argsort(scores)[::-1][:top_k_bm25]

    query_emb = bert_model.encode(query, normalize_embeddings=True)

    cand_embs = doc_embeddings[bm25_idx]
    sim_scores = cosine_similarity([query_emb], cand_embs)[0]

    rerank_idx = bm25_idx[np.argsort(sim_scores)[::-1][:top_k_final]]
    return df.iloc[rerank_idx]

#Wrap 3 model vào
def search_bm25(query):
    tokens = tokenize_bm25(query)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:10]
    return df.iloc[top_idx]


def search_bm25_bert(query):
    return rerank_with_bert(
        query,
        bert_model=bert,
        doc_embeddings=doc_embeddings
    )


def search_bm25_bert_ft(query):
    return rerank_with_bert(
        query,
        bert_model=bert_ft,
        doc_embeddings=doc_embeddings_ft
    )

