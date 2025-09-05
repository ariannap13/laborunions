# Prediction of frame usage on online data

1. We prepare the data for fine-tuning robertabase as a language model with ``./adjust_file_for-finetuning.py``.
2. We fine-tuned a robertabase model on our unions FB posts (``./finetune_roberta.py``).
3. We split our data into train and test sets in ``./split_train_test.py``.
4. We trained the fine-tuned robertabase model on the training set and evaluate it on the test set, following a multi-label task (see ``./train_finetuned_roberta_frames.py`` and ``./inference_finetuned_roberta_frames.py``).
5. Once the performance is checked, we run the model on the entire dataset (using ``./inference_finetuned_roberta_frames.py``), obtaining a prediction dataset as ``../data/predictions_framing_all-data.csv``.

