import torch, time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

text = [
    "Brevity is the soul of wit.",
    "Amor, ch'a nullo amato amar perdona.",
    "Comment allez-vous?"
]

model_ckpt = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


for i in range(50):
    if i == 1:
        start = time.perf_counter()

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    print(inputs["input_ids"].shape)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    preds = torch.softmax(logits, dim=-1)
print(f"Each iteration finished within: {(time.perf_counter() - start) / 49}")
# Map raw predictions to languages
id2lang = model.config.id2label
vals, idxs = torch.max(preds, dim=1)
print({id2lang[k.item()]: v.item() for k, v in zip(idxs, vals)})
