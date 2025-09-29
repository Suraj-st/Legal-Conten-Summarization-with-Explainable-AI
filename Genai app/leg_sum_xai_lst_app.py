import streamlit as st
import spacy
from openai import OpenAI
from lime.lime_text import LimeTextExplainer
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key = api_key)

# Load spaCy model for sentence splitting
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.set_page_config(page_title="Legal Summarizer with XAI", layout="wide")
st.title("⚖️ Legal Document Summarizer with Explainable AI")
st.write("This app simplifies complex legal documents using GPT models and explains the summary if you choose XAI options.")

# Input area
document_text = st.text_area("Paste your legal document text here:", height=250)

# Summarization button

if st.button("Summarize"):
    if document_text.strip() == "":
        st.warning("⚠️ Please enter a legal document before summarizing.")
    else:
        with st.spinner("Summarizing... Please wait."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal assistant AI that simplifies complex legal "
                            "documents into clear, concise, and easy-to-understand summaries. "
                            "Ensure the meaning is preserved while removing unnecessary jargon."
                        )
                    },
                    {"role": "user", "content": document_text}
                ],
                temperature=0.5,
                max_tokens=800,
                top_p=1
            )
            summary = response.choices[0].message.content.strip()

        # Save summary in session_state

        st.session_state["summary"] = summary
        st.session_state["document_text"] = document_text


# Always show summary if exists
if "summary" in st.session_state:
    st.subheader("📄 Simplified Legal Summary")
    st.success(st.session_state["summary"])


    # XAI OPTIONS
    st.markdown("---")
    st.subheader("⚙️ Explainable AI Options")

    use_xai = st.checkbox("Enable Explainable AI (XAI)?")

    if use_xai:
        # Radio Button
        xai_choice = st.radio(
            "Select the explanation method:",
            "Sentence-level Importance", #("Sentence-level Importance", "LIME")
            index=0
        )

        if st.button("XAI Importance"):
            summary = st.session_state["summary"]
            document_text = st.session_state["document_text"]

            if xai_choice == "Sentence-level Importance":

                # Sentence Importance
                st.subheader("🔎 Sentence-level Importance")
                doc = nlp(summary)
                sentences = [sent.text.strip() for sent in doc.sents]

                importance_scores = [len(sent.split()) for sent in sentences]
                max_score = max(importance_scores) if importance_scores else 1

                for i, sent in enumerate(sentences):
                    importance = importance_scores[i] / max_score
                    st.markdown(
                        f"<div style='background-color:rgba(255,165,0,{importance}); padding:8px; border-radius:5px; margin:5px 0;'>"
                        f"{sent} <br><i>Importance Score: {importance_scores[i]}</i></div>",
                        unsafe_allow_html=True
                    )

            elif xai_choice == "LIME":

                # LIME Explanation
                st.subheader("🧠 LIME Explanation for Summary Generation")


                def predict_fn(texts):
                    outputs = []
                    for t in texts:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system",
                                 "content": "Summarize the following legal document into one short sentence."},
                                {"role": "user", "content": t}
                            ],
                            temperature=0,
                            max_tokens=60
                        )
                        outputs.append(resp.choices[0].message.content)
                    # Convert to length-based "target" and reshape for LIME
                    return np.array([len(o.split()) for o in outputs]).reshape(-1, 1)


                explainer = LimeTextExplainer(class_names=["Summary Length"])
                exp = explainer.explain_instance(
                    document_text,
                    predict_fn,
                    num_features=10,
                    labels=(0,)  # specify first column
                )

                st.write("**Top tokens influencing the summary:**")
                for word, weight in exp.as_list(label=0):
                    color = "green" if weight > 0 else "red"
                    st.markdown(f"- <span style='color:{color}'>**{word}**</span>: {weight:.3f}",
                                unsafe_allow_html=True)

    # Optional download
    st.download_button(
        label="💾 Download Summary",
        data=st.session_state["summary"],
        file_name="legal_summary_with_explanations.txt",
        mime="text/plain"
    )