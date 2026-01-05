import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import warnings
import torch

warnings.filterwarnings("ignore")

model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model_name, return_all_scores=True)

max_length = 512
stride = 256  # overlap

def predict_long_text(text):
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"][0]

    # If text is short, classify normally
    if len(input_ids) <= max_length:
        out = classifier(text)[0]
        return {d["label"]: d["score"] for d in out}

    # Build chunks directly as token tensors
    chunk_input_ids = []
    attention_masks = []
    for i in range(0, len(input_ids), stride):
        chunk_ids = input_ids[i:i + max_length]
        if len(chunk_ids) < 10:
            break
        # pad to max_length for batching
        pad_len = max_length - len(chunk_ids)
        chunk_ids = torch.cat([chunk_ids, torch.full((pad_len,), tokenizer.pad_token_id)])
        chunk_input_ids.append(chunk_ids)

        mask = torch.cat([torch.ones(len(chunk_ids)-pad_len), torch.zeros(pad_len)])
        attention_masks.append(mask)

    # stack into batch
    batch = {
        "input_ids": torch.stack(chunk_input_ids),
        "attention_mask": torch.stack(attention_masks)
    }

    # Run model 
    with torch.no_grad():
        outputs = classifier.model(**batch)
        logits = outputs.logits  # shape (num_chunks, num_labels)

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Aggregate by mean
    labels = classifier.model.config.id2label
    agg = {}
    for label_id, label_name in labels.items():
        agg[label_name] = probs[:, label_id].mean()

    return agg


posts_data_topredict = pd.read_csv("../data/fb_data_with_predictions.csv") # version without emotion predictions
posts_data_topredict = posts_data_topredict[['id', 'text']]

emotion_results = []
for idx, row in tqdm(posts_data_topredict.iterrows(), total=len(posts_data_topredict)):
    text = row["text"]
    if isinstance(text, str) and len(text.strip()) > 0:
        scores = predict_long_text(text)
        scores["id"] = row["id"]
        scores["text"] = text
        emotion_results.append(scores)

emotion_df = pd.DataFrame(emotion_results)
emotion_df.to_csv("../data/fb_data_with_predictions.csv", index=False)