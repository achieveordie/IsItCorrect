import torch
import time
from beta_loader import get_pickle_location, get_data
from custom_model import getCustomModel
from transformers import CamembertTokenizer
from tqdm import tqdm
import gc
from pathlib import Path

# Initialize device to run on and base location and loss function, optimizers
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
tokenizer_location = str(Path.cwd().joinpath('sentencepiece.bpe.model'))
tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
criterion = torch.nn.CrossEntropyLoss()

# --------------Things to change every retrain----------------- #
initial_model_location = str(Path.cwd().joinpath('model').joinpath('model_1.pth'))
NUM_EPOCHS = 10
LR = 5e-05
# ------------------------------- #


# Load the model from the above location #
model = getCustomModel()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(initial_model_location))
else:
    model.load_state_dict(torch.load(initial_model_location, map_location='cpu'))
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
# ------------------------------- #


# Copied from train.py, backprops occurs here
def retrain(epoch, x, actual):
    outputs = model(x["input_ids"], x["attention_mask"])
    optimizer.zero_grad()
    loss = criterion(outputs, actual)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    del outputs, x
    torch.cuda.empty_cache()
    gc.collect()
    return loss


# Copied from train.py, manages (re)training across all epochs
def epochs(num_epochs, trainloader):
    for epoch in tqdm(range(num_epochs)):
        iter_trainloader = iter(trainloader)
        time_start_epoch = time.time()
        for counter, data in enumerate(iter_trainloader):

            data["sentence"] = tokenizer(data["sentence"], padding=True, max_length=512)  # tokenize the sentences in a batch
            data["sentence"]["input_ids"] = list(map(lambda x: x[:512], data["sentence"]["input_ids"]))  # take only first 512 characters(limitation of Camembert)
            data["sentence"]["attention_mask"] = list(map(lambda x: x[:512], data["sentence"]["attention_mask"]))
            data["sentence"]["input_ids"] = torch.tensor(data["sentence"]["input_ids"],
                                                         dtype=torch.long, device=device)
            data["sentence"]["attention_mask"] = torch.tensor(data["sentence"]["attention_mask"],
                                                              device=device)

            data["label"] = data["label"].clone().detach().requires_grad_(False).cuda()

            loss = retrain(epoch, data["sentence"], data["label"])
            del data["sentence"]["input_ids"], data["sentence"]["attention_mask"]

            if counter % 1000 == 0:
                print(loss)

                # To print current stats from gpu, useful to identify memory leaks
                # with open(r"memory\memory_data.txt", "a") as file:
                #     stats = torch.cuda.memory_stats("cuda:0")
                #     file.write(pformat(stats))
                #     file.write("\n")
                # with open(r"memory\memory_summary.txt", "a") as file:
                #     stats = torch.cuda.memory_summary("cuda:0")
                #     file.write(pformat({"counter": counter, "summary": stats}))
                #     file.write("\n")
            del data["sentence"], data["label"]
            torch.cuda.empty_cache()
            gc.collect()

        time_end_epoch = time.time()
        print("Done for Epoch Number {}, with total time as {}".format(epoch+1, time_end_epoch-time_start_epoch))
        save_location = str(Path.cwd().joinpath('model').joinpath('model_'+str(epoch)+'actual.pt'))
        print("Saving model to {}".format(save_location))
        torch.save(model.state_dict(), save_location)


if __name__ == '__main__':
    time_start = time.time()
    tokenzier = CamembertTokenizer.from_pretrained(tokenizer_location)
    pickle_location = get_pickle_location
    trainloader = get_data(get_pickle_location)
    epochs(NUM_EPOCHS, trainloader)
    time_end = time.time()
    print("Total Time for Retraining for {} Epochs is {}".format(NUM_EPOCHS, time_end-time_start))
