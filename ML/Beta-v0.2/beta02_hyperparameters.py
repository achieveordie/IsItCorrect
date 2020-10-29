"""
This contains a method which returns all the hyperparameters required for
training/testing purpose. Value only needs to be changed here.
"""


def get_hps():
    """
    `learning_rate`, `batch_size`, `dropout_rate`: self-explainable.
    `retrain_layers`: Number of last layers from transformer to retrain.
    :return:
    """
    hyperparameters = {
        "learning_rate": 1e-05,
        "batch_size": 2,
        "retrain_layers": 15,
        "dropout_rate": 0.1,
        "epochs": 10
    }
    return hyperparameters
