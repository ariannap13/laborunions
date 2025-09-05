import pandas as pd

# AFLCIO FB posts
df = pd.read_csv("../data/full_set_frames.csv")
df = df[["text"]]

## create a txt file with one text per line
with open("train_set_roberta_finetune.txt", "w") as f:
    for text in df["text"]:
        f.write(text + "\n")
