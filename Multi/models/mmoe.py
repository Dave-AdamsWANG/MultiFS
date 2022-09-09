import torch
from layers import EmbeddingLayer, MultiLayerPerceptron, duplicate


class MMoEModel(torch.nn.Module):
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num
        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x)
        categorical_emb = duplicate(categorical_emb)
        numerical_emb = duplicate(numerical_emb)
        emb = [torch.cat([categorical_emb[i], numerical_emb[i].unsqueeze(1)], 1).view(-1, self.embed_output_dim) for i in range(self.task_num)]
        gate_value = [self.gate[i](emb[i]).unsqueeze(1) for i in range(self.task_num)]
        fea = [torch.cat([self.expert[i](emb[j]).unsqueeze(1) for i in range(self.expert_num)], dim = 1) for j in range(self.task_num)]
        task_fea = [torch.bmm(gate_value[i], fea[i]).squeeze(1) for i in range(self.task_num)]
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results