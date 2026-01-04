import streamlit as st
from web import search_bm25, search_bm25_bert, search_bm25_bert_ft

st.set_page_config(page_title="Amazon IR Search", layout="wide")

st.set_page_config(page_title="Amazon IR", layout="wide")
st.title("🔍 Amazon Product Search")

model_choice = st.selectbox(
    "Choose retrieval model",
    [
        "BM25",
        "BM25 + BERT (Base)",
        "BM25 + BERT (Fine-tuned)"
    ])

query = st.text_input("Enter your search query")

if query:
    if model_choice == "BM25":
        results = search_bm25(query)

    elif model_choice == "BM25 + BERT (Base)":
        results = search_bm25_bert(query)

    elif model_choice == "BM25 + BERT (Fine-tuned)":
        results = search_bm25_bert_ft(query)

    for i, row in results.iterrows():
        st.markdown(f"### {row['product_name']}")
        st.write(row['about_product'])
        st.write(f"⭐ Rating: {row['rating']}")

        #LINK SẢN PHẨM
        st.markdown(f"🔗 [View product on Amazon]({row['product_link']})")
        st.markdown("---")


