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


def get_data(location):
    train = load_pickle(location)
    train = CustomLoader(train)
    trainloader = DataLoader(train, batch_size=32, shuffle=False)

    return trainloader
# for i in iter(trainloader):
#     print(i)
#     break


def get_pickle_location(train=True):
    save_location = r"D:\Datasets\IsItCorrect"
    if train:
        pickle_location = os.path.join(save_location, 'beta_train.pkl')
    else:
        pickle_location = os.path.join(save_location, "beta_validate.pkl")
    return pickle_location
