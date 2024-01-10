import torch
import torch.nn as nn
import torch.nn.functional as F
from CoRex import CoRex
from ChronoProphet import ChronoProphet


class Grad(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    def grad(x, constant):
        return Grad.apply(x, constant)



class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)




class Domain_Classifier(nn.Module):
    def __init__(self, num_class, encode_dim):
            super(Domain_Classifier, self).__init__()
            self.num_class = num_class
            self.encode_dim = encode_dim
            self.fc1 = nn.Linear(self.encode_dim, 144)
            self.fc2 = nn.Linear(144,16)
            self.fc3 = nn.Linear(16, num_class)

    def forward(self, input):
        logits = torch.relu(self.fc1(input))
        logits = torch.relu(self.fc2(logits))
        logits = self.fc3(logits)

        return logits



class CORF(nn.Module):
    def __init__(self,device,num_nodes,dropout,
                 supports,gcn_bool,addaptadj,aptinit,in_dim,seq_length,nhid,kernel_size,blocks,layers,
                 forecast_length,backcast_length, nb_blocks_per_stack, out_dim=12):
        super(CORF, self).__init__()
        self.device = device

        self.CoRex = CoRex(device, num_nodes, dropout=dropout,
                           supports=supports, gcn_bool=gcn_bool,
                           in_dim=in_dim, out_dim=seq_length,
                           residual_channels=nhid,
                           kernel_size=2, blocks=1,
                           layers=1
                           ).to(self.device)
        self.ChronoProphet = ChronoProphet(dropout,supports,gcn_bool,addaptadj,aptinit,seq_length,nhid,kernel_size,blocks,layers,
                               device=device, nb_blocks_per_stack= nb_blocks_per_stack,in_dim=in_dim,forecast_length=forecast_length,
                               backcast_length=backcast_length,thetas_dim=(4,8,12),
                               num_nodes=num_nodes,hidden_layer_units=256).to(self.device)

        self.mid_conv = nn.Conv2d(in_channels=forecast_length,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True).to(self.device)

        self.layer1 = nn.Sequential(nn.Linear(in_features=forecast_length, out_features=forecast_length), nn.ReLU(True)).to(self.device)
        self.layer2 = nn.Sequential(nn.Linear(in_features=forecast_length, out_features=forecast_length), nn.ReLU(True)).to(self.device)
        self.layer3 = nn.Sequential(nn.Linear(in_features=forecast_length, out_features=1)).to(self.device)


    def forward(self,input):
        embedding = self.CoRex(input).to(self.device).transpose(1,3)
        encoder_1 = self.layer1(embedding)
        encoder_2 = self.layer2(encoder_1)
        encoder = self.layer3(encoder_2).transpose(1,3)
        backcast, forecast = self.ChronoProphet(encoder)
        forecast = forecast.to(self.device)
        return embedding, forecast
