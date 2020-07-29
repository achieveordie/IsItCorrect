"""Label.py is a must to have to read from the pickle data"""

import pickle
from pathlib import Path

if __name__ == "__main__":
    with open(Path(r'small_test.pkl'), 'rb') as file:
        data = pickle.load(file)
    for a, b in data.items():
        print(a, b)
