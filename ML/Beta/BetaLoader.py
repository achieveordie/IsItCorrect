import pickle
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
train_location = os.path.join(save_location, "beta_sample_train.pkl")

train = load_pickle(train_location)

train = CustomLoader(train)

trainloader = DataLoader(train, batch_size=2, shuffle=True)
for i in iter(trainloader):
    print(i)
    break