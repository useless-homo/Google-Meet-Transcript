from transformers import pipeline
from keybert import KeyBERT

summarizer = pipeline("summarization")
kw_model = KeyBERT()

def generate_summary(text):
    summary = summarizer(text, max_length=130, min_length=40)
    keywords = kw_model.extract_keywords(text, top_n=5)

    action_items = []
    for line in text.split("."):
        if any(v in line.lower() for v in ["need to", "will", "should", "action"]):
            action_items.append(line.strip())

    return {
        "summary": summary[0]["summary_text"],
        "keywords": keywords,
        "action_items": action_items
    }
