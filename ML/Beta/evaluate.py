import torch
from beta_loader import get_pickle_location, get_data
from custom_model import getCustomModel
from transformers import CamembertTokenizer
import time
from tqdm import tqdm

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


def calculate_f_beta(tp, fp, fn, beta=0.5):
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    return ((1 + beta**2)*precision*recall)/((precision * beta**2) + recall)


def evaluate():
    model = load_model(save_model_location)
    tokenizer = CamembertTokenizer.from_pretrained(tokenizer_location)
    testloader = load_data()
    softmax = torch.nn.Softmax(dim=1)
    iter_loader = iter(testloader)
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0  # true-positive, true-negative, false-positive, false-negatives
    print("Starting Evaluation")
    total = 0
    for data in tqdm(iter_loader):

        data["sentence"] = tokenizer(data["sentence"], padding=True, max_length=512)
        data["sentence"]["input_ids"] = list(map(lambda x: x[:512], data["sentence"]["input_ids"]))
        data["sentence"]["attention_mask"] = list(map(lambda x: x[:512], data["sentence"]["attention_mask"]))
        data["sentence"]["input_ids"] = torch.tensor(data["sentence"]["input_ids"],
                                                     dtype=torch.long, device=cuda0)
        data["sentence"]["attention_mask"] = torch.tensor(data["sentence"]["attention_mask"],
                                                          device=cuda0)

        output = model(data["sentence"]["input_ids"], data["sentence"]["attention_mask"])

        # For all data in 1 batch (Here 2 datasets are present in a single batch)
        for i in range(len(data["label"])):
            total += 1
            output = softmax(output)
            actual = data["label"][i].item()
            pred = torch.argmax(output[i]).item()
            if pred == actual:
                correct += 1
            if actual:  # if 1
                if pred:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred:
                    fp += 1
                else:
                    tn += 1

    print("Percentage of correct predictions: {}".format((correct/total * 100.0)))
    print("F-0.5 value is {}".format(calculate_f_beta(tp, fp, tn, fn)))


if __name__ == "__main__":
    time_start = time.time()
    evaluate()
    time_end = time.time()
    print("Total Time Taken: ", time_end - time_start)
