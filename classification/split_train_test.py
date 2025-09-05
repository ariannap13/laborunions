import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.environ["WANDB_DISABLED"] = "true"
import re

data = pd.read_csv("../data/full_set_frames.csv")

# train test split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# export train and test set
train_set.to_csv(f"../data/train_set_framing.csv", index=False)
test_set.to_csv(f"../data/test_set_framing.csv", index=False)