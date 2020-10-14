# from util_withdownload import loadInitialModel   # if want to use base camembert to load as initial model
from util_without import loadInitialModel     # if want to use local model to load as initial model
import torch


class Camembert(torch.nn.Module):
    def __init__(self, cam_model):
        super(Camembert, self).__init__()
        self.l1 = cam_model
        total = 199
        for i, param in enumerate(cam_model.parameters()):
            if total - i > 5:
                param.requires_grad = False
            else:
                pass
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, 576)
        self.l4 = torch.nn.Dropout(0.2)
        self.l5 = torch.nn.Linear(576, 384)
        self.l6 = torch.nn.Linear(384, 2)

    def forward(self, ids, mask):
        _, output = self.l1(ids, attention_mask=mask)
        output = self.l2(output)
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)
        return output

    def printit(self):
        all_layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]
        for layer in all_layers:
            total = 0
            for param in layer.parameters():
                try:
                    print(param.requires_grad, '\t', param.size())
                    total += 1
                except:
                    print("Not printable")
            print("Total is {}".format(total))


def getCustomModel():
    cam_model = loadInitialModel()
    model = Camembert(cam_model)
    return model


# if __name__ == '__main__':
#     model = getCustomModel()
#     model.printit()
