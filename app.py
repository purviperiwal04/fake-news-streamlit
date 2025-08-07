import torch
from transformers import BertTokenizer, BertForSequenceClassification
from IPython.display import display, Markdown

# ✅ Use public Hugging Face model (no need to upload anything)
MODEL_PATH = "mrm8488/bert-tiny-finetuned-fake-news-detection"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load model from Hugging Face
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict_fake_news(text):
    if not text.strip():
        display(Markdown("⚠️ Please enter valid text."))
        return
    
    # ✅ Preprocess
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # ✅ Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item() * 100

    label = "✅ True News" if prediction == 1 else "❌ **Fake News"
    display(Markdown(f"### Prediction: {label}"))
    display(Markdown(f"Confidence: ⁠ {confidence:.2f}% ⁠"))

# Run
sample_text = input("Enter a news article or headline: ")
predict_fake_news(sample_text)
