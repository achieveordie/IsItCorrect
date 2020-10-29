""" note to self: that I am not loading the label with size 512, variable size now"""
""" Source of error- Labels not of same size and incorrect batch loading of labels"""

import torch
import time
from beta02_loader import get_data
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
data_location = Path(r"sample_train.pkl")


def epochs(num_epochs, trainloader):
    for epoch in tqdm(range(num_epochs)):
        trainloder = iter(trainloader)
        time_start_epoch = time.time()
        for counter, data in enumerate(trainloader):
            data["label"] = [torch.cat([i, i.new_zeros(512 - i.size(0))], 0) for i in data["label"]]
            # data["label"] = torch.tensor(data["label"])
            [print("label\t", index, i.size()) for index, i in enumerate(data["label"])]
            data["sentence"] = tokenizer(data["sentence"], padding=True, max_length=512)
            data["sentence"]["input_ids"] = torch.tensor(data["sentence"]["input_ids"],
                                                         dtype=torch.long, device=device)
            [print("sentence\t", index, i.size()) for index, i in enumerate(data["sentence"]["input_ids"])]

            data["sentence"]["input_ids"] = list(map(lambda x: x[:512], data["sentence"]["input_ids"]))
            data["sentence"]["attention_mask"] = list(map(lambda x: x[:512], data["sentence"]["attention_mask"]))

            data["sentence"]["attention_mask"] = torch.tensor(data["sentence"]["attention_mask"],
                                                              device=device)
            # print("sentence is ", data["sentence"])
            # print("Label is", data["label"])
            # data["label"] = torch.tensor(data["label"], device=device)
            # data["label"] = torch.stack(data["label"])
            loss = train(data["sentence"], data["label"])
            print("For epoch number->{}, data number->{}, loss is->{}"
                  .format(epoch, counter, loss))


def train(x, actual):

    outputs = model(x["input_ids"], x["attention_mask"])
    optimizer.zero_grad()
    loss = criterion(outputs, actual)
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    del outputs, x
    torch.cuda.empty_cache()
    gc.collect()
    return loss


if __name__ == '__main__':
    time_start = time.time()
    tokenizer_location = "sentencepiece.bpe.model"
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    trainloader = get_data(data_location)
    epochs(hparams["epochs"], trainloader)

    time_end = time.time()
    print("Total time ", time_end-time_start)

