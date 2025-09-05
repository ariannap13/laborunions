from simpletransformers.classification import  MultiLabelClassificationModel, ClassificationModel

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import logging
import os
os.environ["WANDB_DISABLED"] = "true"
import re

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
cuda_available = torch.cuda.is_available()

mode = "multi-label" 

labels2ids = {"Diagnostic": 0, "Prognostic": 1, "Motivational": 2, "Community": 3, "Engagement": 4, "Political Endorsement": 5}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.cuda.empty_cache()

train_set = pd.read_csv(f"../data/train_set_framing.csv")

data = train_set.copy()

tot_samples = data.shape[0]

manual_seed = 0

data["labels"] = data.apply(lambda x: [x["Diagnostic"], x["Prognostic"], x["Motivational"], x["Community"], x["Engagement"], x["Political Endorsement"]], axis=1)
output_dir = f"../models/classifier_roberta_{mode}"
data = data[["id", "text", "labels"]]

# shuffle the dataset
data = data.sample(frac=1, random_state=manual_seed).reset_index(drop=True)

num_labels = len(labels2ids)

samples_class0 = data[data["labels"].apply(lambda x: x[0] == 1)].shape[0]
samples_class1 = data[data["labels"].apply(lambda x: x[1] == 1)].shape[0]
samples_class2 = data[data["labels"].apply(lambda x: x[2] == 1)].shape[0]
samples_class3 = data[data["labels"].apply(lambda x: x[3] == 1)].shape[0]
samples_class4 = data[data["labels"].apply(lambda x: x[4] == 1)].shape[0]
samples_class5 = data[data["labels"].apply(lambda x: x[5] == 1)].shape[0]
weights = tot_samples / (np.array([samples_class0, samples_class1, samples_class2, samples_class3, samples_class4, samples_class5]))

num_epochs = 30
train_batch_size = 32

model_args = {
'num_train_epochs':num_epochs,
'fp16': True,
"use_early_stopping": True,
"output_dir": output_dir,
"overwrite_output_dir": True,
"manual_seed": manual_seed,
"save_eval_checkpoints": False,
"save_steps": -1,
"train_batch_size": train_batch_size,
"save_model_every_epoch": False,
"logging_steps": 50,
}   
model = MultiLabelClassificationModel('roberta', "../models/roberta_finetuned",
    args=model_args,
    pos_weight = list(weights),
    use_cuda=cuda_available,
    num_labels=num_labels,
)    

model.train_model(data)
