import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import nltk
import shap
from lime.lime_text import LimeTextExplainer
import pandas as pd
import html
import json
import os

# nltk.download('punkt')
# nltk.download('punkt_tab')


# 1. Load models and set device
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summarizer_path = "./legal-summarizer-final"
    sum_tokenizer = AutoTokenizer.from_pretrained(summarizer_path, local_files_only=True)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_path, local_files_only=True).to(device)

    paraphraser_path = "Vamsi/T5_Paraphrase_Paws"
    para_tokenizer = AutoTokenizer.from_pretrained(paraphraser_path)
    para_model = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_path).to(device)

    # para_model_name = "tuner007/pegasus_paraphrase"  # better for fluency
    # para_tokenizer = AutoTokenizer.from_pretrained(para_model_name)
    # para_model = AutoModelForSeq2SeqLM.from_pretrained(para_model_name).to(device)

    return sum_tokenizer, sum_model, para_tokenizer, para_model, device

sum_tokenizer, sum_model, para_tokenizer, para_model, device = load_models()


# 2. Core summarization and paraphrasing logic

def summarize_text(text: str) -> str:
    inputs = sum_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    output_ids = sum_model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=5,
        length_penalty=1.0,
        repetition_penalty=2.0,
        early_stopping=True,
    )
    return sum_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def paraphrase_text(text: str) -> str:
    input_text = f"simplify this into plain English: {text} </s>"
    inputs = para_tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    output_ids = para_model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=5,
        length_penalty=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=False,
    )
    return para_tokenizer.decode(output_ids[0], skip_special_tokens=True)


# 3. Chunking Logic and Sentence-Level Importance

def chunk_text(text: str, chunk_size=800, overlap=100) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def sentence_importance(text: str, summary: str) -> list:
    sentences = nltk.sent_tokenize(text)
    importance = []
    for sent in sentences:
        score = sum([1 for word in sent.split() if word in summary])
        importance.append((sent, score))
    total = sum(score for _, score in importance) + 1e-6
    normalized = [(sent, score / total) for sent, score in importance]
    return normalized

def summarize_long_legal_text(text: str) -> tuple:
    chunks = chunk_text(text)
    explanations = []
    summarized_chunks = []

    for chunk in chunks:
        chunk_summary = summarize_text(chunk)
        chunk_easy = paraphrase_text(chunk_summary)
        chunk_sent_importance = sentence_importance(chunk, chunk_summary)
        explanations.append({
            "chunk": chunk,
            "summary": chunk_summary,
            "easy_summary": chunk_easy,
            "sentence_importance": chunk_sent_importance
        })
        summarized_chunks.append(chunk_summary)

    # combined_summary = " ".join(summarized_chunks)
    # final_easy_summary = paraphrase_text(combined_summary)

    easy_chunks = [paraphrase_text(chunk_summary) for chunk_summary in summarized_chunks]
    final_easy_summary = " ".join(easy_chunks)

    return easy_chunks, final_easy_summary, explanations


# 4. LIME Explanation

def explain_with_lime(text: str, model, tokenizer):
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    def predict_fn(texts):
        preds = pipe(texts, max_length=50)
        # Generate dummy "probabilities" based on length of output
        return np.array([[len(pred["generated_text"]) / 100.0, 1 - len(pred["generated_text"]) / 100.0] for pred in preds])

    explainer = LimeTextExplainer(class_names=["important", "not_important"])
    exp = explainer.explain_instance(text, predict_fn, num_features=10)
    return exp.as_list()



# 5. Highlighting

def highlight_sentences(sentences_with_scores, threshold=0.05):
    html_text = ""
    for sent, score in sentences_with_scores:
        if score > threshold:
            html_text += f"<mark>{html.escape(sent)}</mark> "
        else:
            html_text += html.escape(sent) + " "
    return html_text.strip()

def highlight_all_sentences(sentences_with_scores, threshold=0.05):
    html_text = ""
    for sent, score in sentences_with_scores:
        score_display = f" <sub>({score:.3f})</sub>"  # show score in smaller font
        if score > threshold:
            html_text += f"<mark>{html.escape(sent)}</mark>{score_display} "
        else:
            html_text += f"{html.escape(sent)}{score_display} "
    return html_text.strip()

def highlight_imp_sentences(sentences_with_scores, threshold=0.05):
    html_text = ""
    for sent, score in sentences_with_scores:
        if score > threshold:
            html_text += f"<mark>{html.escape(sent)}</mark> <b>[{score:.3f}]</b> "
        else:
            html_text += html.escape(sent) + " "
    return html_text.strip()


def display_sentence_importance(sentences_with_scores):
    st.subheader("🔎 Sentence-level Importance")
    for sent, score in sentences_with_scores:
        # pick background color intensity based on score
        if score >= 0.2:
            bg_color = "#FFA726"  # darker orange for important
        else:
            bg_color = "#FFCC80"  # lighter orange for less important

        st.markdown(
            f"""
            <div style="background-color:{bg_color}; 
                        padding:12px; 
                        border-radius:10px; 
                        margin-bottom:8px;">
                <p style="margin:0; font-size:15px; font-weight:500;">
                    {html.escape(sent)}
                </p>
                <p style="margin:0; font-size:13px; font-style:italic;">
                    Importance Score: {score:.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# 6. Streamlit UI

st.set_page_config(page_title="Legal Summarizer + Explainable AI", layout="wide")
st.title("📄 Legal Document Summarizer with Explainable AI")

with st.expander("ℹ️ Instructions"):
    st.markdown("""
        - Paste your legal document.
        - Summary and simplified version will be generated.
        - Enable Explainable AI for interpretation.
        - Sentence importance and SHAP/LIME-style explanations supported.
    """)

legal_text = st.text_area("Enter Legal Text Here", height=300)
enable_xai = st.toggle("Enable Explainable AI")
explain_method = st.radio("XAI Method", ["Sentence-Level", "LIME"], index=0) if enable_xai else None

if st.button("🔍 Summarize"):
    if not legal_text.strip():
        st.warning("Please enter a legal document.")
    else:
        with st.spinner("Analyzing..."):
            inputs = sum_tokenizer(legal_text, return_tensors="pt", truncation=True)
            if inputs["input_ids"].shape[1] <= 1024:
                summary = summarize_text(legal_text)
                easy_summary = paraphrase_text(summary)
                explanations = [{
                    "chunk": legal_text,
                    "summary": summary,
                    "easy_summary": easy_summary,
                    "sentence_importance": sentence_importance(legal_text, summary)
                }]
            else:
                summary, easy_summary, explanations = summarize_long_legal_text(legal_text)

        st.subheader("🧾 Summary")
        st.success(summary)

        st.subheader("🗣️ Easy-to-Understand Summary")
        st.info(easy_summary)

        if enable_xai:
            st.subheader("🧠 Explainability Insights")
            for idx, expl in enumerate(explanations):
                st.markdown(f"**Chunk {idx + 1}**")
                if explain_method == "Sentence-Level":
                    sent_html = highlight_sentences(expl["sentence_importance"])
                    st.markdown(sent_html, unsafe_allow_html=True)
                elif explain_method == "LIME":
                    try:
                        lime_exp = explain_with_lime(expl["chunk"], sum_model, sum_tokenizer)
                        for word, score in lime_exp:
                            st.markdown(f"- **{word}** → `{round(score, 3)}`")
                    except Exception as e:
                        st.error(f"LIME failed: {str(e)}")

        # Export
        export_filename = "xai_explanations.json"
        with open(export_filename, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=2)

        with open(export_filename, "rb") as f:
            st.download_button("📁 Download Explanations (JSON)", f, file_name=export_filename)