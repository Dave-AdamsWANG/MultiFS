import torch
import torch.nn as nn
from utils import Weights
import numpy as np

class EMB(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x).transpose(1,2)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x





class select_MLP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.num = len(args.field_dims)
        self.dec_f = args.decide_f # decide function
        if not args.field_dims[-1]:
            self.num = self.num - 1
            self.dec_f = 0
        self.emb = EMB(args.field_dims[:self.num],args.embed_dim)
        self.mlp = MultiLayerPerceptron(input_dim=len(args.field_dims)*args.embed_dim,
                                        embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)
        self.weights = Weights(args.softmax_type,args.field_dims)
        self.stage = 0
        self.decision = None


    def forward(self, field):
        field = self.emb(field)
        if not self.dec_f:
            field = torch.cat([field,torch.ones_like(field[:,:,None,0])],-1) 
        if self.stage ==1:
            weight = self.weights()[:,-1]
            field = field * weight
        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        return torch.sigmoid(res.squeeze(1))







class MLP(nn.Module):
    def __init__(self, args, decision):
        super().__init__()
        self.stage = 2
        field_dims = args.field_dims
        self.decision = [i*decision[i] for i in range(len(decision)) if decision[i]]
        field_dims = [field_dims[i] for i in self.decision]
        self.emb = EMB(field_dims,args.embed_dim)
        self.device = args.device
        num = sum(decision)
        self.mlp = MultiLayerPerceptron(input_dim=num*args.embed_dim,
                                        embed_dims=args.mlp_dims, output_layer=True, dropout=args.dropout)

    def forward(self,field):
        field = self.emb(field[:,self.decision])
        input_mlp = field.flatten(start_dim=1)
        res = self.mlp(input_mlp)
        return torch.sigmoid(res.squeeze(1))



