from simpletransformers.language_modeling import  LanguageModelingModel
import torch
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
cuda_available = torch.cuda.is_available()

trainfile='../data/train_set_roberta_finetune.txt' # this is created from the 
outdir='../models/roberta_finetuned/'

model_args = {
        'num_train_epochs':5,
        "use_early_stopping": True,
        "output_dir": outdir,
        "overwrite_output_dir": True,
        "manual_seed": 42,
        "save_eval_checkpoints": False,
        "save_steps": -1,
        "train_batch_size": 128,
        "n_gpu": 1,
}

model = LanguageModelingModel("roberta", "roberta-base", args=model_args,use_cuda=cuda_available)
model.train_model(trainfile)