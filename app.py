import os
import time
import csv
import json
import re
import traceback
import streamlit as st
from openai import OpenAI
from embeddings_utils import build_store_from_textfile, embed_texts, rerank_by_overlap
import numpy as np
import pandas as pd

st.set_page_config(page_title="Club RAG Agent (Hybrid)", layout="centered")
st.write("DEBUG: CWD ->", os.getcwd())

API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "6"))

client = OpenAI(api_key=API_KEY, base_url=BASE)

st.sidebar.markdown("## Controls")
allow_fallback = st.sidebar.checkbox("Allow LLM fallback", value=True)
rebuild = st.sidebar.button("Build / Rebuild Index")
uploaded = st.sidebar.file_uploader("Upload club_data.txt", type=["txt"])
download_feedback = st.sidebar.button("Refresh feedback preview")

if uploaded:
    with open("/content/uploaded_club_data.txt", "wb") as f:
        f.write(uploaded.getbuffer())
    data_path = "/content/uploaded_club_data.txt"
else:
    data_path = "/content/club_data.txt" if os.path.exists("/content/club_data.txt") else "club_data.txt"

if "store" not in st.session_state or rebuild:
    try:
        with st.spinner("Building index..."):
            st.session_state["store"] = build_store_from_textfile(data_path)
            st.success("Index ready.")
    except Exception as e:
        st.error(f"Index build failed: {e}")

FEEDBACK_DIR = "/content"
os.makedirs(FEEDBACK_DIR, exist_ok=True)
FEEDBACK_FH = f"{FEEDBACK_DIR}/feedback.csv"
FEEDBACK_LOCK = f"{FEEDBACK_DIR}/feedback.lock"
FEEDBACK_JSONL = f"{FEEDBACK_DIR}/feedback.jsonl"
FEEDBACK_ERROR = f"{FEEDBACK_DIR}/feedback.error.log"

from filelock import FileLock, Timeout

def _append_feedback_row(row, fh=FEEDBACK_FH, lock_fh=FEEDBACK_LOCK, timeout_secs=8):
    lock = FileLock(lock_fh)
    try:
        with lock.acquire(timeout=timeout_secs):
            write_header = not os.path.exists(fh) or os.path.getsize(fh) == 0
            with open(fh, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            return True
    except Exception as e:
        with open(FEEDBACK_JSONL, "a", encoding="utf-8") as jf:
            jf.write(json.dumps({"ts": time.time(), "row": row, "error": str(e)}) + "\n")
        with open(FEEDBACK_ERROR, "a", encoding="utf-8") as elog:
            elog.write(traceback.format_exc() + "\n")
        return False

def show_feedback_preview():
    st.sidebar.markdown("### Collected feedback")
    if os.path.exists(FEEDBACK_FH):
        try:
            df = pd.read_csv(FEEDBACK_FH)
            st.sidebar.write(f"Rows: {len(df)}")
            st.sidebar.dataframe(df.tail(6))
            st.sidebar.download_button("Download feedback.csv", df.to_csv(index=False).encode("utf-8"), "feedback.csv")
        except Exception as e:
            st.sidebar.write("Could not read feedback.csv:", e)
    else:
        st.sidebar.write("No feedback yet.")

show_feedback_preview()

def tokenize(text):
    return re.findall(r"\w+", (text or "").lower())

def sent_support(answer_text, ctx_tokens):
    sents = [s.strip() for s in re.split(r"[.!?]\s*", (answer_text or "")) if s.strip()]
    if not sents:
        return 0.0
    supported = sum(1 for s in sents if set(tokenize(s)) & ctx_tokens)
    return supported / len(sents)

STRICT_SYSTEM_PROMPT = (
    "You MUST answer ONLY using the provided CONTEXT. "
    "If the answer is NOT in the CONTEXT, reply EXACTLY: 'I don't know based on the provided club data.'"
)

FALLBACK_SYSTEM_PROMPT = (
    "Use context if possible; otherwise use world knowledge and answer concisely."
)

st.title("Club Information RAG Agent — Hybrid mode" if allow_fallback else "Strict RAG mode")
question = st.text_input("Ask about the club:")

if st.button("Ask") and question.strip():
    if "store" not in st.session_state:
        st.error("Index not ready.")
    else:
        store = st.session_state["store"]
        q_emb = embed_texts([question])[0]

        results = store.search(q_emb, top_k=TOP_K)
        reranked = rerank_by_overlap(question, results, alpha=0.20)

        st.subheader("Retrieved Passages")
        combined_context = ""
        for md, sim, overlap_score, combined in reranked:
            txt = md.get("text") if isinstance(md, dict) else str(md)
            st.write(f"**(sim {sim:.3f})** — {txt[:600]}")
            combined_context += "\n\n" + txt

        user_prompt = f"CONTEXT:\n{combined_context}\n\nQUESTION: {question}"

        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.0
            )
            llm_answer = resp.choices[0].message["content"]
        except:
            llm_answer = ""

        ans_tokens = tokenize(llm_answer)
        ctx_tokens = set(tokenize(combined_context))

        overlap_frac = sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens) if ans_tokens else 0
        sent_frac = sent_support(llm_answer, ctx_tokens)

        sims = [float(x[1]) for x in reranked]
        top_sim = max(sims) if sims else 0
        k = min(3, len(sims))
        topk_mean_sim = sum(sorted(sims, reverse=True)[:k]) / k if k else 0

        accept = (
            overlap_frac >= 0.30 or
            sent_frac >= 0.30 or
            top_sim >= 0.55 or
            topk_mean_sim >= 0.40
        )

        used_fallback = False

        if accept:
            final_answer = llm_answer
        else:
            if allow_fallback:
                resp_fb = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.0
                )
                final_answer = resp_fb.choices[0].message["content"]
                used_fallback = True
            else:
                final_answer = "I don't know based on the provided club data."

        st.subheader("Answer")
        if used_fallback:
            st.info("Answer from general knowledge.")
        st.write(final_answer)

        def show_feedback_ui(question, answer, combined_context, reranked_chunks, fallback_flag=False):
            st.markdown("---")
            st.write("**Was this answer helpful?**")
            with st.form("feedback_form", clear_on_submit=True):
                label = st.radio("Select:", ["Correct", "Partially correct", "Incorrect"], horizontal=True)
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    mapping = {"Correct":"correct","Partially correct":"partial","Incorrect":"incorrect"}
                    chosen = mapping[label]

                    used = []
                    for md, *_ in reranked_chunks[:4]:
                        if isinstance(md, dict) and md.get("chunk_id") is not None:
                            used.append(str(md["chunk_id"]))
                        else:
                            txt = md.get("text") if isinstance(md, dict) else str(md)
                            used.append(str(abs(hash(txt)) % 1_000_000_000))

                    row = {
                        "ts": int(time.time()),
                        "query": question.replace("\n"," "),
                        "answer": answer.replace("\n"," "),
                        "combined_context": combined_context.replace("\n"," "),
                        "used_chunks": ";".join(used),
                        "label": chosen,
                        "fallback_used": str(bool(fallback_flag))
                    }

                    ok = _append_feedback_row(row)

                    if ok:
                        st.success("Feedback saved.")
                        st.experimental_rerun()
                    else:
                        st.warning("Write failed.")
                        st.experimental_rerun()

        show_feedback_ui(question, final_answer, combined_context, reranked, used_fallback)

        with st.expander("Debug info"):
            st.write({
                "overlap": overlap_frac,
                "sent_support": sent_frac,
                "top_sim": top_sim,
                "topk_mean": topk_mean_sim,
                "used_fallback": used_fallback
            })
