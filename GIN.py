import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


# Step 1: build GIN model, we should also test SGC

class GIN(nn.Module):
    """
    Note: batch normalization can prevent divergence maybe, take care of this later. 
    """
    def __init__(self,  nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), 
                                       act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.transform(x) # weird as normalization is applying to all ndoes in database
        # maybe a better way is to normalize the mean of each graph, and then apply tranformation
        # to each groups * 
        #embed = self.pooling(x, batch)
        #std = torch.sqrt(self.pooling((x - embed[batch])**2, batch))
        #graph_embeds = [embed]
        #graph_stds = [std]
        # can I also record the distance to center, which is the variance?
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)
            #embed = self.pooling(x, batch) # embed is the center of nodes
            #std = torch.sqrt(self.pooling((x - embed[batch])**2, batch))
            #graph_embeds.append(embed)
            #graph_stds.append(std)

        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch==g]
            emb_list.append(emb)
        #graph_embeds = torch.stack(graph_embeds)
        #graph_stds = torch.stack(graph_stds)

        return emb_list