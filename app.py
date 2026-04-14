import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "/content/drive/MyDrive/dataset/Distilbert_model"

st.set_page_config(
    page_title="ABSA Dashboard",
    layout="wide",
    page_icon="📊"
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

nlp = spacy.load("en_core_web_sm", disable=["ner"])

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
def get_aspect_context(review, aspect):
    review_lower = review.lower()

    for connector in [" but ", " however ", " although "]:
        if connector in review_lower:
            parts = review_lower.split(connector)
            for part in parts:
                if aspect in part:
                    return part.strip()

    doc = nlp(review)
    for sent in doc.sents:
        if aspect in sent.text.lower():
            return sent.text.strip()

    return review

def analyze_review(review):

    if review.strip() == "":
        return pd.DataFrame({"Message": ["Please enter a review."]}), None

    aspects = [chunk.text.lower() for chunk in nlp(review).noun_chunks][:3]

    if not aspects:
        return pd.DataFrame({"Message": ["No aspects detected."]}), None

    results = []

    for aspect in aspects:

        context = get_aspect_context(review, aspect)
        text = context + " [SEP] " + aspect

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=96
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

        results.append({
            "Aspect": aspect,
            "Context Used": context,
            "Sentiment": label_map[pred],
            "Confidence": round(confidence, 3)
        })

    result_df = pd.DataFrame(results)

    fig, ax = plt.subplots()
    result_df['Sentiment'].value_counts().plot(
        kind='bar',
        ax=ax,
        color=['red', 'gray', 'green']
    )
    ax.set_title("Aspect Sentiment Distribution")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()

    return result_df, fig

# BENCHMARK TABLE

results_df = pd.DataFrame({
    "Model": ["DistilBERT", "SVM", "Logistic Regression"],
    "Macro F1": [0.723197, 0.591823, 0.569401], 
    "Accuracy": [0.8924542, 0.851779, 0.778256]
})


st.title("📊 Aspect-Based Sentimental Analysis")
st.markdown("""
MSc Data Science & Artificial Intelligence Project

✅ Clause-level aspect extraction
✅ Weighted transformer training
✅ Proper ABSA evaluation
""")

# LIVE ANALYSIS


st.header("🔍 Live ABSA Analysis")

review_input = st.text_area(
    "Enter Product Review",
    height=150,
    placeholder="Example: The battery is good but screen is bad."
)

if st.button("Analyze Review"):
    result_df, fig = analyze_review(review_input)
    st.dataframe(result_df)
    if fig:
        st.pyplot(fig)

# BENCHMARK RESULTS


st.header("📈 Model Benchmark Results")

st.dataframe(results_df)

fig2, ax2 = plt.subplots()
ax2.bar(results_df["Model"], results_df["Macro F1"])
ax2.set_title("Macro F1 Comparison")
ax2.set_ylabel("F1 Score")
plt.xticks(rotation=30)
st.pyplot(fig2)

st.header("⚠ Error Analysis")

st.markdown("""
**Observed Findings:**

- Clause-level supervision improved minority class detection.
- Weighted loss reduced majority class bias.
- Contrast sentences (e.g., "good but bad") are handled correctly.
- Implicit sentiment remains challenging.

**Future Work:**

- Domain-adaptive pretraining
- Larger balanced dataset
- Joint aspect extraction and sentiment modelling
""")
