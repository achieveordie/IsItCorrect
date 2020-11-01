"""
This contains a method which returns all the hyperparameters required for
training/testing purpose. Value only needs to be changed here.
"""


def get_hps():
    """
    `learning_rate`, `batch_size`, `dropout_rate`: self-explainable.
    `retrain_layers`: Number of last layers from transformer to retrain.
    :return: <dict> of hyperparameters.

    The following are the result from training on sample data, useful to decide hps.
    (loss: 0.1786, time: 74.71) -> hyperparameters = {
        "learning_rate": 1e-05,
        "batch_size": 2,
        "retrain_layers": 15,
        "dropout_rate": 0.1,
        "epochs": 10
    }
    (loss: 0.1445, time: 86.09 sec) -> hyperparameters = {
        "learning_rate": 1e-05,
        "batch_size": 4,
        "retrain_layers": 15,
        "dropout_rate": 0.1,
        "epochs": 20
    }
    (loss: 0.1588, time: 87.41) -> hyperparameters = {
        "learning_rate": 1e-05,
        "batch_size": 2,
        "retrain_layers": 30,
        "dropout_rate": 0.1,
        "epochs": 10
    }
    (loss: 0.1554, time:98.72) -> hyperparameters = {
        "learning_rate": 1e-05,
        "batch_size": 2,
        "retrain_layers": 60,
        "dropout_rate": 0.1,
        "epochs": 10
    }
    (loss: 0.0527, time: 294.80) -> hyperparameters = {
        "learning_rate": 1e-05,
        "batch_size": 2,
        "retrain_layers": 15,
        "dropout_rate": 0.1,
        "epochs": 40
    }
    (loss:0.0485, time: 92.29) -> hyperparameters = {
        "learning_rate": 5e-05,
        "batch_size": 2,
        "retrain_layers": 40,
        "dropout_rate": 0.1,
        "epochs": 10
    }
    (loss: 0.049, time: 80.18) -> hyperparameters = {
        "learning_rate": 5e-05,
        "batch_size": 2,
        "retrain_layers": 20,
        "dropout_rate": 0.1,
        "epochs": 10
    }

    Now using lr_scheduling
    (loss: 0.7056, time: 83.30) -> hyperparameters = {
        "learning_rate": 5e-05,
        "batch_size": 2,
        "retrain_layers": 40,
        "dropout_rate": 0.1,
        "epochs": 10,
        "step_size": 10,  # Number of epochs to step down lr
        "gamma": 0.1  # The multiplicative factor by which lr steps down after every `step_size`
    }
    """
    hyperparameters = {
        "learning_rate": 5e-05,
        "batch_size": 2,
        "retrain_layers": 40,
        "dropout_rate": 0.1,
        "epochs": 10,
        "step_size": 10,  # Number of epochs to step down lr
        "gamma": 0.1  # The multiplicative factor by which lr steps down after every `step_size`
    }
    return hyperparameters
