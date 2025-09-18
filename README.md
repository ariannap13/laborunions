This folder contains all the code and data to reproduce the analyses of the paper *"Framing Unionization on Facebook: Communication around Representation Elections in the United States"*.

All analyses are performed with Python 3.8.16. The list of required package versions can be found in `requirements.txt`.

Here is a brief description of the sub-folders (further details can be found in the `README.md` files inside the folders):

1. `process-data` – Scripts for processing data from the NLRB and Facebook.  
2. `classification` – Code for training and testing the discourse frames classifier.  
3. `union-activity_analysis` – Scripts to analyze unions’ online activity and usage of discourse frames over election events.  
4. `data` – Useful data files.

---

## Pre-trained Models

The models used in this project are available on Hugging Face:

- **[RoBERTa fine-tuned on union Facebook posts](https://huggingface.co/ariannap22/roberta_finetuned_fb_unions)** – Language model fine-tuned on union-related language.  
- **[RoBERTa multi-label discourse frames classifier](https://huggingface.co/ariannap22/roberta_finetuned_multilabel_unions)** – Classifies posts into the following frames: Diagnostic, Prognostic, Motivational, Community, Engagement, Political Endorsement.

You can load these models in Python using `simpletransformers`:

```python
from simpletransformers.classification import MultiLabelClassificationModel
import torch

cuda_available = torch.cuda.is_available()

model = MultiLabelClassificationModel(
    'roberta',
    "ariannap22/roberta_finetuned_multilabel_unions",
    use_cuda=cuda_available
)