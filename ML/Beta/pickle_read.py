import pickle
from pprint import pprint


pickle_location = r"D:\Datasets\IsItCorrect\beta_sample_train_small.pkl"
with open(pickle_location, 'rb') as file:
    data = pickle.load(file)
print(data[0]["sentence"])
print(data[1]["sentence"])
