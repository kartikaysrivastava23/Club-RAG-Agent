import os
import streamlit as st
from openai import OpenAI
from embeddings_utils import build_store_from_textfile, embed_texts, rerank_by_overlap
import numpy as np

API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "6"))

client = OpenAI(api_key=API_KEY, base_url=BASE)

st.set_page_config(page_title="Club RAG Agent", layout="centered")
st.title("Club Information RAG Agent")

st.sidebar.markdown("## Controls")
uploaded = st.sidebar.file_uploader("Upload club_data.txt (optional)", type=["txt"])
rebuild = st.sidebar.button("Build / Rebuild Index")

if uploaded:
    with open("uploaded_club_data.txt","wb") as f:
        f.write(uploaded.getbuffer())
    data_path = "uploaded_club_data.txt"
else:
    data_path = "club_data.txt"

if 'store' not in st.session_state or rebuild:
    try:
        with st.spinner("Building index..."):
            st.session_state['store'] = build_store_from_textfile(data_path)
            st.success("Index ready.")
    except Exception as e:
        st.error(f"Index build failed: {e}")

question = st.text_input("Ask about the club:")

if st.button("Ask") and question.strip():
    if 'store' not in st.session_state:
        st.error("Index not ready.")
    else:
        store = st.session_state['store']
        q_emb = embed_texts([question])[0]

        results = store.search(q_emb, top_k=TOP_K)
        reranked = rerank_by_overlap(question, results, alpha=0.20)

        st.subheader("Retrieved Passages")
        combined_context = ""
        for md, sim, overlap_score, combined in reranked:
            st.write(
                f"**(sim {sim:.3f} | overlap {overlap_score:.2f} | combined {combined:.3f})** — "
                f"{md['text'][:600]}{'...' if len(md['text']) > 600 else ''}"
            )
            combined_context += "\n\n" + md['text']

        system_prompt = (
            "You must answer ONLY using the provided CONTEXT. "
            "If information is missing, reply: 'I don't know based on the provided club data.'"
        )

        user_prompt = (
            f"CONTEXT:\n{combined_context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using ONLY the context."
        )

        with st.spinner("Querying model..."):
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.0,
            )

        answer = (
            resp.choices[0].message["content"]
            if hasattr(resp.choices[0].message, "get")
            else resp.choices[0].message.content
        )

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Raw response"):
            st.json(resp)
