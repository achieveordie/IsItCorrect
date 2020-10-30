import pickle
from torch.utils.data import Dataset, DataLoader
from beta02_hyperparameters import get_hps
from Label import make_label
import torch

hparams = get_hps()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def get_data(location):
    """
    The method which is to be called from outside the file, handles the rest.
    :param location: location of pickle file
    :return: an instance of DataLoader
    """
    data = load_pickle(location)
    data = CustomLoaderBeta02(data)
    return DataLoader(data, batch_size=hparams["batch_size"], shuffle=False)


def load_pickle(location):
    """
    Load a given pickle file.
    :param location: the address of the pickle file which is to be read
    :return: Deserialized object:
    {0: {'label':<list>, 'sentence':<string>},
     1 : {'label':<list>, 'sentence':<string>}...}
    """
    with open(location, 'rb') as file:
        return pickle.load(file)


class CustomLoaderBeta02(Dataset):
    """
    Custom Loader, inheriting Dataset and implementing two abstract classes.
    """
    def __init__(self, data):
        self.data = data
        # map(lambda x: make_label(x["label"]), self.data)
        # for i in range(len(self.data)):
        #     self.data[i]["label"] = make_label(self.data[i]["label"])

    def __getitem__(self, item):
        self.data[item]["label"] = torch.tensor(self.data[item]["label"], device=device)
        return self.data[item]

    def __len__(self):
        return len(self.data)
