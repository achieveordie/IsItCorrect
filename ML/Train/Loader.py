import pickle
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.insert(1, r"D:\IsItCorrect\Labeling Scripts")

from Label import Label


class CustomLoader(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def load_pickle(location):
    with open(location, 'rb') as file:
        return pickle.load(file)


save_location = r"D:\Datasets\IsItCorrect"
train_location = os.path.join(save_location, "sample_train.pkl")
test_location = os.path.join(save_location, "sample_test.pkl")

train = load_pickle(train_location)
test = load_pickle(test_location)

train = CustomLoader(train)
test = CustomLoader(test)

trainloader = DataLoader(train, batch_size=2)
testloader = DataLoader(test, batch_size=2)
