import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import types
def snip_forward_linear(self, x):
    return [F.linear(x, self.weight * self.weight_mask[i], self.bias) for i in range(2)]
def snip_forward_embedding(self,x):
    return [F.embedding(x,self.weight * self.weight_mask[i]) for i in range(2)] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

forward_mapping_dict = {
    'Linear': snip_forward_linear,
    'Embedding': snip_forward_embedding
}
def weight_reset(m):
    for i in m.named_children():
        if i[0] not in ['embedding','numerical_layer']: 
            for j in i[1].modules():
                if type(j).__name__ == 'Linear':
                    j.reset_parameters()

class Prunner:
    def __init__(self, model, criterion, dataloader):
        self.model = copy.deepcopy(model).to(device)
        self.prun_model = copy.deepcopy(model).to(device)
        self.criterion = criterion.to(device)
        self.dataloader = dataloader
        self.variance_scaling_init()
        self.update_forward_pass()

    def apply_hook(self, masks):
        n_layers = list(filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.numerical_layer.modules()))
        c_layers = list(filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.embedding.modules()))
        def apply_masking(mask):
            def hook(weight):
                return weight * mask
            return hook
        layers = n_layers + c_layers
        for layer, mask in zip(layers, masks):
            assert layer.weight.shape == mask.shape
            layer.weight.data = layer.weight.data * mask
            layer.weight.register_hook(apply_masking(mask))
        
        
    def prun(self, compression_factor=0.5, num_batch_sampling=1):
        weight_reset(self.prun_model)
        grads_list = self.compute_grads(num_batch_sampling)
        masks = []
        for i in range(grads_list[0].shape[0]): #task_num
            gl = [grads_list[j][i] for j in range(len(grads_list))]
            grads = torch.cat([torch.flatten(grad) for grad in gl])
            keep_params = int((1 - compression_factor) * len(grads))
            values, idxs = torch.topk(grads / grads.sum(), keep_params, sorted=True)
            threshold = values[-1]
            masks.append([(grad / grads.sum() > threshold).float() for grad in gl])
        for i in range(len(masks)):
            if i == 0:
                mask = masks[i]
            else: 
                mask[0] = torch.max(masks[i][0],mask[0])
                mask[1] = torch.max(masks[i][1],mask[1])
        self.apply_hook(mask)
        return self.prun_model, masks

    def compute_grads(self, num_batch_sampling=1):
        moving_average_grad_list = [[]] * self.model.task_num
        for i, (c_data, n_data, labels) in enumerate(self.dataloader):
            if i == num_batch_sampling:
                break
            c_data, n_data, labels = c_data.to(device), n_data.to(device),labels.to(device)
            out = self.model(c_data,n_data)
            loss_list = [self.criterion(out[i], labels[:, i].float()) for i in range(labels.size(1))]
            loss = 0
            for item in loss_list:
                loss += item
            self.model.zero_grad()
            loss.backward()
            grads_list = []
            for layer in self.model.numerical_layer.modules():
                if type(layer).__name__ in forward_mapping_dict:
                    grads_list.append(torch.abs(layer.weight_mask.grad))
            for layer in self.model.embedding.modules():
                if type(layer).__name__ in forward_mapping_dict:
                    grads_list.append(torch.abs(layer.weight_mask.grad))
            if i == 0:
                moving_average_grad_list = grads_list
            else:
                moving_average_grad_list = [((mv_avg_grad * i) + grad) / (i + 1)
                                        for mv_avg_grad, grad in zip(moving_average_grad_list, grads_list)]
            return moving_average_grad_list

    def variance_scaling_init(self):
        for layer in self.model.numerical_layer.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.weight_mask = nn.Parameter(torch.ones_like(torch.cat([layer.weight.unsqueeze(0)]*self.model.task_num,0)).to(device))
                #nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
                
        for layer in self.model.embedding.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.weight_mask = nn.Parameter(torch.ones_like(torch.cat([layer.weight.unsqueeze(0)]*self.model.task_num,0)).to(device))
                #nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
    
    def update_forward_pass(self):
        for layer in self.model.numerical_layer.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)
        for layer in self.model.embedding.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy
# import types
# def snip_forward_linear(self, x):
#     return F.linear(x, self.weight * self.weight_mask, self.bias)
# def snip_forward_embedding(self,x):
#     return F.embedding(x,self.weight * self.weight_mask)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# forward_mapping_dict = {
#     'Linear': snip_forward_linear,
#     'Embedding': snip_forward_embedding
# }
# def weight_reset(m):
#     for i in m.named_children():
#         if i[0] not in ['embedding','numerical_layer']: 
#             for j in i[1].modules():
#                 if type(j).__name__ == 'Linear':
#                     j.reset_parameters()

# class Prunner:

#     def __init__(self, model, criterion, dataloader):
#         self.model = copy.deepcopy(model).to(device)
#         self.prun_model = copy.deepcopy(model).to(device)
#         self.criterion = criterion.to(device)
#         self.dataloader = dataloader
#         self.variance_scaling_init()
#         self.update_forward_pass()

#     def apply_hook(self, masks):
#         n_layers = list(filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.numerical_layer.modules()))
#         c_layers = list(filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.embedding.modules()))
#         def apply_masking(mask):
#             def hook(weight):
#                 return weight * mask
#             return hook
#         layers = n_layers + c_layers
#         for layer, mask in zip(layers, masks):
#             assert layer.weight.shape == mask.shape
#             layer.weight.data = layer.weight.data * mask
#             layer.weight.register_hook(apply_masking(mask))
        
        
#     def prun(self, compression_factor=0.5, num_batch_sampling=1):
#         weight_reset(self.prun_model)
#         grads, grads_list = self.compute_grads(num_batch_sampling)
#         keep_params = int((1 - compression_factor) * len(grads))
#         values, idxs = torch.topk(grads / grads.sum(), keep_params, sorted=True)
#         threshold = values[-1]
#         masks = [(grad / grads.sum() > threshold).float() for grad in grads_list]
#         self.apply_hook(masks)
#         return self.prun_model, masks

#     def compute_grads(self, num_batch_sampling=1):
#         moving_average_grads = 0
#         for i, (c_data, n_data, labels) in enumerate(self.dataloader):
#             if i == num_batch_sampling:
#                 break
#             c_data, n_data, labels = c_data.to(device), n_data.to(device),labels.to(device)
#             out = self.model(c_data,n_data)
#             loss_list = [self.criterion(out[i], labels[:, i].float()) for i in range(labels.size(1))]
#             loss = 0
#             for item in loss_list:
#                 loss += item
#             self.model.zero_grad()
#             loss.backward()
#             grads_list = []
#             for layer in self.model.numerical_layer.modules():
#                 if type(layer).__name__ in forward_mapping_dict:
#                     grads_list.append(torch.abs(layer.weight_mask.grad))
#             for layer in self.model.embedding.modules():
#                 if type(layer).__name__ in forward_mapping_dict:
#                     grads_list.append(torch.abs(layer.weight_mask.grad))
#             grads = torch.cat([torch.flatten(grad) for grad in grads_list])
#             if i == 0:
#                 moving_average_grads = grads
#                 moving_average_grad_list = grads_list
#             else:
#                 moving_average_grads = ((moving_average_grads * i) + grads) / (i + 1)
#                 moving_average_grad_list = [((mv_avg_grad * i) + grad) / (i + 1)
#                                             for mv_avg_grad, grad in zip(moving_average_grad_list, grads_list)]
#         return moving_average_grads, moving_average_grad_list

#     def variance_scaling_init(self):
#         for layer in self.model.numerical_layer.modules():
#             if type(layer).__name__ in forward_mapping_dict:
#                 layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
#                 #nn.init.xavier_normal_(layer.weight)
#                 layer.weight.requires_grad = False
                
#         for layer in self.model.embedding.modules():
#             if type(layer).__name__ in forward_mapping_dict:
#                 layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
#                 #nn.init.xavier_normal_(layer.weight)
#                 layer.weight.requires_grad = False
    
#     def update_forward_pass(self):
#         for layer in self.model.numerical_layer.modules():
#             if type(layer).__name__ in forward_mapping_dict:
#                 layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)
#         for layer in self.model.embedding.modules():
#             if type(layer).__name__ in forward_mapping_dict:
#                 layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)