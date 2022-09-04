import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Weights(torch.nn.Module):

    def __init__(self, softmax_type,field_dims):
        super().__init__()
        self.field_size = len(field_dims)
        self.candidate   = 2
        if field_dims[-1] == 0: self.candidate = 1
        initial_deep = np.ones((self.field_size, self.candidate))/self.candidate
        self.deep_weights = torch.nn.Parameter(torch.from_numpy(initial_deep), requires_grad=True)
        self.softmax_type = softmax_type
        self.tau = 1.0

    def forward(self,):
        if self.tau > 0.01:
            self.tau -= 0.00005
        # print(f'self.tau={round(self.tau, 5)}')

        if self.softmax_type == 0:
            return F.softmax(self.deep_weights, dim=1)
        elif self.softmax_type == 1:
            return F.softmax(self.deep_weights/self.tau, dim=1)
        elif self.softmax_type == 2:
            if self.candidate==1:
                return F.softmax(self.deep_weights, dim=0)*len(self.deep_weights)/2
            return F.gumbel_softmax(self.deep_weights, tau=self.tau, hard=False, dim=-1)
        else:
            print('No such softmax_type'); print('TAU={}'.format(TAU)); quit()