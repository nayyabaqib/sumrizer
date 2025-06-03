import streamlit as st
from transformers import pipeline
import textwrap

# ------------------ App Configuration ------------------
st.set_page_config(page_title="Smart Summarizer", layout="centered")
st.title(" Smart Text Summarizer")
st.markdown("Summarize long English articles and documents with AI ")

# ------------------ Load Model ------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ------------------ Helper Function: Chunk Long Text ------------------
def split_text(text, max_words=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count <= max_words:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_len = word_count
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks

# ------------------ Text Input ------------------
text = st.text_area(" Paste your article or long text here:", height=300)

# ------------------ Summary Button ------------------
if st.button(" Summarize"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary... "):
            try:
                chunks = split_text(text)
                summary_results = []
                for chunk in chunks:
                    summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                    summary_results.append(summary)

                final_summary = ' '.join(summary_results)

                st.success(" Summary Generated:")
                st.text_area(" Summarized Text", final_summary, height=200)

            except Exception as e:
                st.error(f" Error: {e}")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Built with  BART model and Streamlit | Works offline after first use.")
