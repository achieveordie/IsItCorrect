from beta02_utils import load_initial_model
import torch
from beta02_hyperparameters import get_hps

hparams = get_hps()


def get_custom_model():
    """
    The method which is to be called from other files, handles the rest.
    :return: returns the instance of custom model.
    """
    cam_model = load_initial_model()
    return Camembert(cam_model)


class Camembert(torch.nn.Module):
    """
    The definition of the custom model, last 15 layers of Camembert will be retrained
    and then a fcn to 512 (the size of every label).
    """
    def __init__(self, cam_model):
        super(Camembert, self).__init__()
        self.l1 = cam_model
        total_layers = 199
        for i, param in enumerate(cam_model.parameters()):
            if total_layers - i > hparams["retrain_layers"]:
                param.requires_grad = False
            else:
                pass
        self.l2 = torch.nn.Dropout(hparams["dropout_rate"])
        self.l3 = torch.nn.Linear(768, 512)
        self.activation = torch.nn.Sigmoid()

    def forward(self, ids, mask):
        _, output = self.l1(ids, attention_mask=mask)
        output = self.l2(output)
        output = self.l3(output)
        return self.activation(output)
