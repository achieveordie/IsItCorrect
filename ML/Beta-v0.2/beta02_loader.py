import pickle
from torch.utils.data import Dataset, DataLoader


def get_data(location):
    """
    The method which is to be called from outside the file, handles the rest.
    :param location: location of pickle file
    :return: an instance of DataLoader
    """
    data = load_pickle(location)
    data = CustomLoaderBeta02(data)
    return DataLoader(data, batch_size=16, shuffle=False)


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

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
