import torch
import time
from beta02_loader import load_pickle, get_data
from beta02_custom_model import get_custom_model
from beta02_hyperparameters import get_hps
from transformers import CamembertTokenizer
from tqdm import tqdm
from pprint import pformat
import gc
from pathlib import Path

# Setting up some basic settings.
hparams = get_hps()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Training will take place on", device)

model = get_custom_model().to(device)
model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["learning_rate"])
save_model_location = Path(r"D:\Datasets\IsItCorrect\model\beta_02")

if __name__ == '__main__':
    time_start = time.time()
    tokenizer_location = "sentencepiece.bpe.model"
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    time_end = time.time()
    print("Total time ", time_end-time_start)

