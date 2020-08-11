from util import loadInitialModel
import torch


class Camembert(torch.nn.Module):
    def __init__(self, cam_model):
        super(Camembert, self).__init__()
        self.l1 = cam_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)

    def forward(self, ids, mask):
        _, output = self.l1(ids, attention_mask=mask)
        output = self.l2(output)
        output = self.l3(output)
        return output


def getCustomModel():
    cam_model = loadInitialModel()
    model = Camembert(cam_model)
    return model
