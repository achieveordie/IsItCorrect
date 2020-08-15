import torch
import time
from beta_loader import get_pickle_location, get_data
from custom_model import getCustomModel
from transformers import CamembertTokenizer

cuda0 = torch.device('cuda:0')
model = getCustomModel().to(cuda0)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)


def train(epoch, x, actual):
    model.train()
    ids = x["input_ids"]
    attention_mask = x["attention_mask"]
    outputs = model(ids, attention_mask)
    optimizer.zero_grad()
    loss = criterion(outputs, actual)
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def epochs(num_epochs, trainloader):
    time_start = time.time()
    for epoch in range(num_epochs):
        iter_trainloader = iter(trainloader)
        counter = 0
        for data in iter_trainloader:
            data["sentence"] = tokenizer(data["sentence"], padding=True)
            data["sentence"]["input_ids"] = torch.tensor(data["sentence"]["input_ids"],
                                                         dtype=torch.long, device=cuda0)
            data["sentence"]["attention_mask"] = torch.tensor(data["sentence"]["attention_mask"],
                                                              device=cuda0)
            data["label"] = torch.tensor(data["label"], device=cuda0)
            loss = train(epoch, data["sentence"], data["label"])
            counter += 1
            if counter % 10 == 0:
                print(loss)
        print("Done for Epoch number {}".format(epoch))
    time_end = time.time()
    print("Total Time Taken: ", time_end-time_start)


if __name__ == '__main__':
    tokenizer_location = r"D:\Datasets\camembert-base\sentencepiece.bpe.model"
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    pickle_location = get_pickle_location()
    trainloader = get_data(pickle_location)
    epochs(50, trainloader)
