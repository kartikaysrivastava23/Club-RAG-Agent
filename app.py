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

st.set_page_config(page_title="Club Information RAG Agent", layout="centered")
st.write("DEBUG: CWD ->", os.getcwd())

API_KEY = os.environ.get("OPENROUTER_API_KEY")
BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "6"))

client = OpenAI(api_key=API_KEY, base_url=BASE)

st.sidebar.markdown("## Controls")
rebuild = st.sidebar.button("Build / Rebuild Index")
uploaded = st.sidebar.file_uploader("Upload club_data.txt (optional)", type=["txt"])
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

try:
    from filelock import FileLock, Timeout
except Exception:
    FileLock = None
    Timeout = Exception

def _append_feedback_row(row, fh=FEEDBACK_FH, lock_fh=FEEDBACK_LOCK, timeout_secs=8):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(fh)) or ".", exist_ok=True)
    except:
        pass

    if FileLock is None:
        try:
            write_header = not os.path.exists(fh) or os.path.getsize(fh) == 0
            with open(fh, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except:
                    pass
            return True
        except Exception as e:
            try:
                with open(FEEDBACK_JSONL, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps({"ts": time.time(), "row": row, "error": str(e)}) + "\n")
            except:
                pass
            try:
                with open(FEEDBACK_ERROR, "a", encoding="utf-8") as elog:
                    elog.write(traceback.format_exc() + "\n")
            except:
                pass
            return False

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
                try:
                    os.fsync(f.fileno())
                except:
                    pass
            return True
    except Timeout:
        try:
            with open(FEEDBACK_ERROR, "a", encoding="utf-8") as elog:
                elog.write(f"{int(time.time())} TIMEOUT acquiring lock {fh}\n")
        except:
            pass
        return False
    except Exception as e:
        try:
            with open(FEEDBACK_JSONL, "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"ts": time.time(), "row": row, "error": str(e)}) + "\n")
        except:
            pass
        try:
            with open(FEEDBACK_ERROR, "a", encoding="utf-8") as elog:
                elog.write(traceback.format_exc() + "\n")
        except:
            pass
        return False

try:
    import pandas as pd
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
except Exception:
    st.sidebar.write("Feedback preview unavailable.")

def tokenize(text):
    return re.findall(r"\w+", (text or "").lower())

def sent_support(answer_text, ctx_tokens):
    sents = [s.strip() for s in re.split(r"[.!?]\s*", (answer_text or "")) if s.strip()]
    if not sents:
        return 0.0
    supported = 0
    for s in sents:
        toks = set(tokenize(s))
        if len(toks & ctx_tokens) > 0:
            supported += 1
    return supported / len(sents)

STRICT_SYSTEM_PROMPT = (
    "You MUST answer ONLY using the provided CONTEXT. "
    "You are NOT allowed to use outside knowledge. "
    "If the answer is NOT fully present in the CONTEXT, reply EXACTLY: 'I don't know based on the provided club data.' "
    "Do NOT add, invent, infer, or guess any information."
)

st.title("Club Information RAG Agent")
question = st.text_input("Ask about the club (answers must come from club_data.txt):")

if st.button("Ask") and question.strip():
    if "store" not in st.session_state:
        st.error("Index not ready.")
    else:
        store = st.session_state["store"]
        q_emb = embed_texts([question])[0]

        results = store.search(q_emb, top_k=TOP_K)
        reranked = rerank_by_overlap(question, results, alpha=0.20)

        st.subheader("Retrieved Passages (top-ranked)")
        combined_context = ""
        for md, sim, overlap_score, combined in reranked:
            txt = md.get("text") if isinstance(md, dict) else str(md)
            st.write(f"**(sim {sim:.3f} | overlap {overlap_score:.2f} | combined {combined:.3f})** — {txt[:600]}{'...' if len(txt)>600 else ''}")
            combined_context += "\n\n" + txt

        user_prompt = f"CONTEXT:\n{combined_context}\n\nQUESTION: {question}"
        try:
            with st.spinner("Querying model..."):
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.0
                )
            try:
                llm_answer = resp.choices[0].message["content"]
            except Exception:
                try:
                    llm_answer = resp.choices[0].message.content
                except Exception:
                    llm_answer = str(resp)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            llm_answer = ""

        ans_tokens = tokenize(llm_answer)
        ctx_tokens = set(tokenize(combined_context))
        overlap_frac = 0.0 if len(ans_tokens)==0 else sum(1 for t in ans_tokens if t in ctx_tokens)/float(len(ans_tokens))
        sent_frac = sent_support(llm_answer, ctx_tokens)

        top_sim = 0.0
        topk_mean_sim = 0.0
        try:
            sims = [float(x[1]) for x in reranked]
            if sims:
                top_sim = max(sims)
                k = min(3, len(sims))
                topk_mean_sim = sum(sorted(sims, reverse=True)[:k]) / float(k)
        except Exception:
            top_sim = 0.0
            topk_mean_sim = 0.0

        TH_overlap = 0.30
        TH_sent_frac = 0.30
        TH_top_sim = 0.55
        TH_topk_mean = 0.40

        accept = (overlap_frac >= TH_overlap) or (sent_frac >= TH_sent_frac) or (top_sim >= TH_top_sim) or (topk_mean_sim >= TH_topk_mean)

        if accept:
            final_answer = llm_answer
        else:
            final_answer = "I don't know based on the provided club data."

        st.subheader("Answer")
        st.write(final_answer)

        def show_feedback_ui_inner(question, answer, combined_context, reranked_chunks):
            st.markdown("---")
            st.write("**Was this answer helpful?**")
            with st.form("feedback_form", clear_on_submit=True):
                try:
                    label = st.radio("Select:", ["Correct", "Partially correct", "Incorrect"], horizontal=True)
                except Exception:
                    label = st.radio("Select:", ["Correct", "Partially correct", "Incorrect"])
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    mapping = {"Correct":"correct","Partially correct":"partial","Incorrect":"incorrect"}
                    chosen = mapping.get(label, "none")
                    used = []
                    try:
                        for md, *_ in reranked_chunks[:4]:
                            if isinstance(md, dict) and md.get("chunk_id") is not None:
                                used.append(str(md["chunk_id"]))
                            else:
                                txt = md.get("text") if isinstance(md, dict) else str(md)
                                used.append(str(abs(hash(txt)) % 1_000_000_000))
                    except Exception:
                        used = []

                    row = {
                        "ts": int(time.time()),
                        "query": (question or "").replace("\n"," "),
                        "answer": (answer or "").replace("\n"," "),
                        "combined_context": (combined_context or "").replace("\n"," "),
                        "used_chunks": ";".join(used),
                        "label": chosen,
                        "fallback_used": str(False)
                    }

                    try:
                        dbg_path = "/content/feedback_debug.log"
                        with open(dbg_path, "a", encoding="utf-8") as dbg:
                            dbg.write(f"[{time.time()}] DEBUG: about to write row: {row}\n")
                            dbg.flush()
                    except Exception:
                        pass

                    try:
                        with open("/content/feedback_last_submit.txt", "w", encoding="utf-8") as mf:
                            mf.write(str(time.time()))
                    except Exception:
                        pass

                    try:
                        ok = _append_feedback_row(row)
                    except Exception:
                        ok = False

                    try:
                        with open("/content/feedback_debug.log", "a", encoding="utf-8") as dbg:
                            dbg.write(f"[{time.time()}] DEBUG: _append_feedback_row returned: {ok}\n")
                            dbg.flush()
                    except Exception:
                        pass

                    abs_path = os.path.abspath(FEEDBACK_FH)
                    if ok:
                        st.success(f"Thanks — feedback saved to {abs_path}")
                        st.experimental_rerun()
                    else:
                        st.warning(f"Saved to fallback or failed.")
                        st.experimental_rerun()

        show_feedback_ui_inner(question, final_answer, combined_context, reranked)

        with st.expander("Debug: grounding info"):
            st.write({
                "overlap_frac": round(overlap_frac,3),
                "sent_frac": round(sent_frac,3),
                "top_sim": round(top_sim,3),
                "topk_mean_sim": round(topk_mean_sim,3),
                "accepted_strict": bool(accept)
            })
            st.write({"question": question})
            st.write({"retrieved_count": len(reranked)})
            if len(reranked) > 0:
                try:
                    top_md = reranked[0][0]
                    preview = (top_md.get("text","")[:400] if isinstance(top_md, dict) else str(top_md)[:400])
                    st.write({"top_chunk_preview": preview})
                except:
                    pass
            st.markdown("**Raw LLM response (strict prompt)**")
            st.write(llm_answer)

        with st.expander("Raw response"):
            try:
                st.json(resp)
            except:
                st.write(str(resp))
