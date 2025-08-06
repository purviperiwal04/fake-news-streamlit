# force rebuild
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ✅ Load model and tokenizer
MODEL_PATH = "model"  # Your saved model folder
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(device)  # ✅ Load on CPU/MPS properly
    return tokenizer, model

tokenizer, model = load_model()

st.title("📰 Fake News Detection App")
st.write("Enter a news article headline or text to check if it's **Fake** or **True**.")

user_input = st.text_area("Enter your news text here:")

if st.button("Predict"):
    if user_input.strip():
        # ✅ Preprocess input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # ✅ Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item() * 100

        label = "✅ True News" if prediction == 1 else "❌ Fake News"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text to predict.")
