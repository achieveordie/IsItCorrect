import torch
import time
from beta_loader import get_pickle_location, get_data
from custom_model import getCustomModel
from transformers import CamembertTokenizer
from tqdm import tqdm
from pprint import pformat
import gc


cuda0 = torch.device('cuda:0')
model = getCustomModel().to(cuda0)
model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)
save_model_location = r"D:\Datasets\IsItCorrect\model\model_actual.pth"


def train(epoch, x, actual):

    # ids = x["input_ids"]
    # attention_mask = x["attention_mask"]
    # print(torch.cuda.memory_allocated(), "\t", torch.cuda.max_memory_allocated())
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


def epochs(num_epochs, trainloader):
    for epoch in tqdm(range(num_epochs)):
        iter_trainloader = iter(trainloader)
        time_start_epoch = time.time()
        for counter, data in enumerate(iter_trainloader):

            data["sentence"] = tokenizer(data["sentence"], padding=True)
            data["sentence"]["input_ids"] = torch.tensor(data["sentence"]["input_ids"],
                                                         dtype=torch.long, device=cuda0)
            data["sentence"]["attention_mask"] = torch.tensor(data["sentence"]["attention_mask"],
                                                              device=cuda0)

            #data["label"] = torch.tensor(data["label"], device=cuda0)
            data["label"] = data["label"].clone().detach().requires_grad_(False).cuda()

            loss = train(epoch, data["sentence"], data["label"])
            del data["sentence"]["input_ids"], data["sentence"]["attention_mask"]
            #print("----------------SPOT-2------------------------------")
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj["data"])):
            #             print(type(obj), obj.size())
            #     except:
            #         pass

            if counter % 100 == 0:
                print(loss)
                with open(r"memory\memory_data.txt", "a") as file:
                    stats = torch.cuda.memory_stats("cuda:0")
                    file.write(pformat(stats))
                    file.write("\n")
                with open(r"memory\memory_summary.txt", "a") as file:
                    stats = torch.cuda.memory_summary("cuda:0")
                    file.write(pformat({"counter": counter, "summary": stats}))
                    file.write("\n")
            del data["sentence"], data["label"]
            torch.cuda.empty_cache()
            gc.collect()
            # print("----------------SPOT-3------------------------------")

        time_end_epoch = time.time()
        print("Done for Epoch number {}, with time {}".format(epoch + 1, time_end_epoch-time_start_epoch))
    print("Saving Model to {}".format(save_model_location))
    torch.save(model.state_dict(), save_model_location)


if __name__ == '__main__':
    time_start = time.time()
    tokenizer_location = r"D:\Datasets\camembert-base\sentencepiece.bpe.model"
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    pickle_location = get_pickle_location()
    trainloader = get_data(pickle_location)
    epochs(25, trainloader)
    time_end = time.time()
    print("Total Time Taken For training: ", time_end - time_start)

    # print("Now begins the evaluation: ")
    # time_eval_start = time.time()


