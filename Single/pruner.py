import torch 
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torchfm.model.dfm import DeepFactorizationMachineModel

class Pruner(nn.Module):
    def __init__(self, origin_model, criterion, dataloader):
        super(Pruner, self).__init__()
        self.model = deepcopy(origin_model.model)
        self.emb =  deepcopy(origin_model.emb)
        self.device =self.emb.embedding.weight.device
        self.origin_model = OriginModel(self.emb,self.model,mask=1)
        self.criterion = criterion
        self.dataloader = dataloader

    def forward(self, compression_factor=0.5, num_batch_sampling=1):
        grads, grads_list = self.compute_grads(num_batch_sampling)
        keep_params = int((1 - compression_factor) * len(grads))
        values, idxs = torch.topk(grads / grads.sum(), keep_params, sorted=True)
        threshold = values[-1]
        masks = [(grad / grads.sum() > threshold).float() for grad in grads_list]
        return masks
    
    def compute_grads(self, num_batch_sampling=1):
        moving_average_grads = 0
        for i, (data, labels) in enumerate(self.dataloader):
            if i == num_batch_sampling:
                break
            x, labels = data.to(self.device), labels.to(self.device)
            out = self.origin_model(x)
            loss = self.criterion(out, labels.float())
            self.origin_model.zero_grad()
            loss.backward()
            grads_list = []
            grads_list.append(torch.abs(self.origin_model.mask.grad))
            grads = torch.cat([torch.flatten(grad) for grad in grads_list])
            if i == 0:
                moving_average_grads = grads
                moving_average_grad_list = grads_list
            else:
                moving_average_grads = ((moving_average_grads * i) + grads) / (i + 1)
                moving_average_grad_list = [((mv_avg_grad * i) + grad) / (i + 1)
                                            for mv_avg_grad, grad in zip(moving_average_grad_list, grads_list)]
        return moving_average_grads, moving_average_grad_list


from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork

def get_model(name,field_dims,embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=embed_dim)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=embed_dim)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=embed_dim, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlpmlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=embed_dim, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=embed_dim, num_layers=3, mlp_dims=(16, 8), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=embed_dim, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=embed_dim, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=embed_dim, mlp_dims=(16, 8), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=embed_dim, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=embed_dim, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=embed_dim, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=embed_dim, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


class PrunedModel(nn.Module):
    def __init__(self,origin_model, model_name,field_dims,masks):
        super(PrunedModel, self).__init__()
        self.d = masks.sum(1).int().tolist()
        self.field_dims = []
        for i in range(len(field_dims)):
            if self.d[i]>0: 
                self.field_dims.append(field_dims[i])
        self.dmax = max(self.d)
        self.emb = deepcopy(origin_model.emb)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.model = get_model(model_name,self.field_dims,self.dmax)
        self.embedding = nn.ModuleList(nn.Embedding.from_pretrained(self.emb.embedding.weight[self.offsets[i]:self.offsets[i]+field_dims[i],masks[i]>0]) for i in range(len(field_dims)))
        self.linear = nn.ModuleList(nn.Linear(self.d[i],self.dmax) for i in range(len(field_dims)))

    def forward(self,x):
        embed_list = []
        decision = []
        for i in range(x.shape[1]):
            if self.d[i]:
                embed_list.append(self.linear[i](self.embedding[i](x[:,i])).unsqueeze(1))
                decision.append(i)
        x_ = torch.cat([x[:,None,i] for i in decision],-1)
        embed_x = torch.cat(embed_list,1)
        return self.model(x_,embed_x)

class OriginModel(nn.Module):
    def __init__(self, emb, model, mask =None):
        super(OriginModel,self).__init__()
        self.emb = emb
        self.model = model
        self.device =self.emb.embedding.weight.device
        self.mask = nn.Parameter(torch.ones([len(self.emb.field_dims), self.emb.embed_dim]).to(self.device))
    
    def forward(self,x):
        embed_x = self.emb(x)
        if self.mask is not None:
            embed_x = embed_x * self.mask
        return self.model(x,embed_x)







    
                
