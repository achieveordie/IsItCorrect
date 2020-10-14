import torch
from beta_loader import get_pickle_location, get_data
from custom_model import getCustomModel
from transformers import CamembertTokenizer
import time

cuda0 = torch.device("cuda:0")

save_model_location = r"D:\Datasets\IsItCorrect\model\model0_actual.pth"
tokenizer_location = r"D:\Datasets\camembert-base\sentencepiece.bpe.model"


def load_model(save_model_location):
    model = getCustomModel().to(cuda0)
    model.load_state_dict(torch.load(save_model_location))
    model.eval()
    return model


def load_data():
    pickle_location = get_pickle_location(train=False)
    testloader = get_data(pickle_location)
    return testloader


def evaluate():
    model = load_model(save_model_location)
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    testloader = load_data()
    softmax = torch.nn.Softmax(dim=1)
    iter_loader = iter(testloader)
    correct = 0
    for data in iter_loader:
        # print("Data going in: ")
        # print(data['sentence'])

        data["sentence"] = tokenizer(data["sentence"], padding=True)
        data["sentence"]["input_ids"] = torch.tensor(data["sentence"]["input_ids"],
                                                     dtype=torch.long, device=cuda0)
        data["sentence"]["attention_mask"] = torch.tensor(data["sentence"]["attention_mask"],
                                                          device=cuda0)
        output = model(data["sentence"]["input_ids"], data["sentence"]["attention_mask"])

        # For all data in 1 batch (Here 2 datasets are present in a single batch)
        for i in range(len(data["label"])):
            output = softmax(output)
            pred = torch.argmax(output[i]).item()
            # print("Label : {}, Prediction: {}, with Probability= {}".format(data["label"][i].item(),
            #                                                                 pred,
            #                                                                 output[i][pred].item()*100.0))
            if pred == data["label"][i].item():
                correct += 1
    print("Percentage of correct predictions: {}".format((correct/len(testloader["label"]) * 100.0)))


if __name__ == "__main__":
    time_start = time.time()
    evaluate()
    time_end = time.time()
    print("Total Time Taken: ", time_end - time_start)