import pickle
import random
from time import time
from typing import Union
from CoRex import CoRex

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy
from torch.optim import Optimizer


class ChronoProphet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 dropout,
                 supports,
                 gcn_bool,
                 adaptadj,
                 aptinit,
                 seq_length,
                 nhid,
                 kernel_size,
                 blocks,
                 layers,
                 device=torch.device('cpu'),
                 stack_types=(TREND_BLOCK,SEASONALITY_BLOCK,GENERIC_BLOCK),
                 nb_blocks_per_stack=1,
                 in_dim=1,
                 forecast_length=12,
                 backcast_length=12,
                 thetas_dim=(4, 8,12),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 num_nodes=207,
                 ):
        super(ChronoProphet, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        self.dropout = dropout
        self.supports = supports
        self.gcn_bool = gcn_bool
        self.adaptadj = adaptadj
        self.aptinit = aptinit
        self.seq_length = seq_length
        self.nhid = nhid
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.layers = layers
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        #print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = ChronoProphet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.num_nodes, self.dropout, self.supports, self.gcn_bool, self.adaptadj,
                                   self.aptinit, self.in_dim, self.seq_length, self.nhid,
                                   self.hidden_layer_units, self.thetas_dim[stack_id],
                                   self.device, self.backcast_length, self.forecast_length, self.kernel_size,
                                   self.blocks, self.layers
                                 )
                self.parameters.extend(block.parameters())
            #print(f'     | -- {block}')
            blocks.append(block)
        return blocks


    @staticmethod
    def select_block(block_type):
        if block_type == ChronoProphet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == ChronoProphet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        #backcast = squeeze_last_dim(backcast)
        forecast = torch.zeros(size=(backcast.size()[0], self.in_dim, self.num_nodes, self.forecast_length,))  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
        # print("------------------------------")
        # print(backcast.shape, forecast.shape)
        # print("-------------------------------")
        return backcast, forecast




def seasonality_model(thetas, t, device):
    p = thetas.size()[1]
    #assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2]).to(device)
    return torch.einsum('ncvl,cw->nwvl', (thetas, S))


def trend_model(thetas, t, device):
    p = thetas.size()[1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor(np.array([t ** i for i in range(p)])).float().to(device)
    return torch.einsum('ncvl,cw->nwvl', (thetas, T))



def linear_space(backcast_length, forecast_length):
    ls = np.arange(-backcast_length, forecast_length, 1) / forecast_length
    b_ls = np.abs(np.flip(ls[:backcast_length]))
    f_ls = ls[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, num_nodes, dropout, supports, gcn_bool, adaptadj, aptinit, in_dim, seq_length, nhid,
                 units, thetas_dim, device, backcast_length, forecast_length, kernel_size, blocks, layers, share_thetas=False):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.device = device
        self.CoRex = CoRex(device, num_nodes, dropout=dropout,
                           supports=supports, gcn_bool=gcn_bool,
                           addaptadj=adaptadj, aptinit=aptinit,
                           in_dim=in_dim, out_dim=seq_length,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16,
                           kernel_size=kernel_size,blocks=blocks,
                           layers=layers
                           )
        
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, forecast_length)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Conv2d(in_channels=seq_length,
                                                            out_channels = thetas_dim,
                                                            kernel_size=(1,1))

        else:
            self.theta_f_fc = nn.Conv1d(in_channels=seq_length,
                                        out_channels= thetas_dim,
                                        kernel_size=(1,1))
            self.theta_b_fc = nn.Conv1d(in_channels=seq_length,
                                        out_channels=thetas_dim,
                                        kernel_size=(1, 1))

    def forward(self, x):
        x = self.CoRex(x.to(self.device))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class SeasonalityBlock(Block):

    def __init__(self, num_nodes, dropout, supports, gcn_bool, adaptadj, aptinit, in_dim, seq_length, nhid,
                 units, thetas_dim, device, backcast_length, forecast_length,kernel_size,blocks,layers, nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(num_nodes, dropout, supports, gcn_bool, adaptadj,
                                                   aptinit, in_dim, seq_length, nhid,
                                                   units, nb_harmonics, device, backcast_length,
                                                   forecast_length, kernel_size, blocks, layers, share_thetas=True)

        else:
            super(SeasonalityBlock, self).__init__(num_nodes, dropout, supports, gcn_bool, adaptadj,
                                                   aptinit, in_dim, seq_length, nhid,
                                                   units, forecast_length, device, backcast_length,
                                                   forecast_length, kernel_size, blocks, layers, share_thetas=True
                                                   )

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        backcast = backcast.transpose(1, 3)
        forecast = forecast.transpose(1, 3)
        # print("--------Seasonality-----")
        # print(forecast.shape)
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, num_nodes, dropout, supports, gcn_bool, adaptadj,
                 aptinit, in_dim, seq_length, nhid,
                 units, thetas_dim, device, backcast_length, forecast_length,kernel_size,blocks,layers, nb_harmonics=None,
                ):
        super(TrendBlock, self).__init__(num_nodes, dropout, supports, gcn_bool, adaptadj,
                                         aptinit, in_dim, seq_length, nhid,
                                         units, thetas_dim, device, backcast_length,
                                         forecast_length,kernel_size,blocks,layers, share_thetas=True,
                                         )

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        # print("------------------")
        # print(x.shape)
        # print(self.theta_b_fc(x).shape)
        # print(self.theta_f_fc(x).shape)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        backcast = backcast.transpose(1, 3)
        forecast = forecast.transpose(1, 3)
        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, num_nodes,dropout,supports, gcn_bool, adaptadj,
                                         aptinit, in_dim, seq_length, nhid,
                                         units, thetas_dim, device, backcast_length,
                                         forecast_length,kernel_size,blocks,layers, share_thetas=True):
        super(GenericBlock, self).__init__(num_nodes, dropout, supports, gcn_bool, adaptadj,
                                         aptinit, in_dim, seq_length, nhid,
                                         units, thetas_dim, device, backcast_length,
                                         forecast_length,kernel_size,blocks,layers, share_thetas=True)

        self.backcast_fc1 = nn.Conv2d(in_channels=backcast_length,
                                     out_channels=6,
                                     kernel_size=(1, 1))
        self.backcast_fc2 = nn.Conv2d(in_channels=6,
                                      out_channels=backcast_length,
                                      kernel_size=(1, 1))
        self.forecast_fc1 = nn.Conv2d(in_channels=forecast_length,
                                     out_channels=6,
                                     kernel_size=(1, 1))
        self.forecast_fc2 = nn.Conv2d(in_channels=6,
                                      out_channels=forecast_length,
                                      kernel_size=(1, 1))

    def forward(self, x):
        x = x.transpose(1, 3)

        backcast1 = self.backcast_fc1(x)  # generic. 3.3.
        forecast1 = self.forecast_fc1(x)  # generic. 3.3.
        backcast = self.backcast_fc2(backcast1)
        forecast = self.forecast_fc2(forecast1)
        backcast.transpose(1, 3)
        forecast.transpose(1, 3)

        return backcast, forecast
