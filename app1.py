import streamlit as st
from ir_models import search_bm25, search_bm25_bert

st.set_page_config(page_title="Amazon IR Search", layout="wide")

st.set_page_config(page_title="Amazon IR", layout="wide")
st.title("🔍 Amazon Product Search")

model_choice = st.sidebar.selectbox(
    "Choose retrieval model",
    ["BM25", "BM25 + BERT", "BM25 + BERT (fine-tuned)"]
)

query = st.text_input("Enter your search query")

if query:
    if model_choice == "BM25":
        results = search_bm25(query)

    elif model_choice == "BM25 + BERT":
        results = search_bm25_bert(query)

    for _, row in results.iterrows():
        st.markdown(f"### {row['product_name']}")
        st.write(row["about_product"])
        st.divider()

