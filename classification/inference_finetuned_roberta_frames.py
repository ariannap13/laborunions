from simpletransformers.classification import  MultiLabelClassificationModel, ClassificationModel

import pandas as pd
import torch
import logging
import os
os.environ["WANDB_DISABLED"] = "true"
import re

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
cuda_available = torch.cuda.is_available()

mode = "multi-label" # "multi-label" or "binary"
target = "test-set" # "test-set" or "full-data"

labels2ids = {"Diagnostic": 0, "Prognostic": 1, "Motivational": 2, "Community": 3, "Engagement": 4, "Political Endorsement": 5}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.cuda.empty_cache()

# Read test set
if target == "test-set":
    test_set = pd.read_csv(f"../data/test_set_framing.csv")
else:
    test_set = pd.read_csv(f"../data/full_set_frames.csv")

data = test_set.copy()

data = data.dropna(subset=["text"])

model = MultiLabelClassificationModel('roberta', f"../models/classifier_roberta_{mode}", use_cuda=cuda_available)

predictions, raw_outputs = model.predict(data["text"].tolist())

# save predictions
data['predictions'] = predictions

data.to_csv(f"../data/predictions_framing_{target}.csv", index=False)
