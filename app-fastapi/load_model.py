from huggingface_hub import hf_hub_download
from flash.text import TextClassifier
import os

current_path = os.getcwd()
REPO_ID = "AndrewTsai0406/bert-based-chinese-law-article-classifier"
MODEL_NAME = 'law-classification-ckiplab-bert-base-chinese.pt'
# MODEL_NAME = 'law-classification-ckiplab-bert-base-chinese.pt'
hf_hub_download(repo_id=REPO_ID, filename=MODEL_NAME, local_dir_use_symlinks=False, local_dir='./models')
REPO_ID = "AndrewTsai0406/bert-based-chinese-jail-sentence-classifier"
MODEL_NAME = 'sentence-classification-ckiplab-bert-base-chinese.pt'
# MODEL_NAME = 'sentence-classification-ckiplab-bert-base-chinese.pt'
hf_hub_download(repo_id=REPO_ID, filename=MODEL_NAME, local_dir_use_symlinks=False, local_dir='./models')