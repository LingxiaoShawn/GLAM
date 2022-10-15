#utils.py

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from mixhop_generator import MixhopGraphGenerator, random_split_counts


class SimpleGraphDataset(InMemoryDataset):
    def __init__(self, name, data_list):
        self.name = name
        self.data, self.slices = self.collate(data_list)

        self.__indices__ = None
        self.transform = None

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


def load_synthetic_data(num_train=500, num_test_inlier=500, num_test_outlier=25, h_inlier=0.7, h_outlier=0.3, n_min = 50, n_max = 150, no_of_tags = 5, type1 = "mixhop", type2 = "mixhop", seed = 1213):
    np.random.seed(seed)
    print('generating data')
    g_list = []
    
    for i in range(num_train+num_test_inlier):
        
        n = np.random.randint(n_min, n_max)
        
        if type1 == "mixhop":
            tag_counts = random_split_counts(n, no_of_tags)
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0)(n, 2, 10, h_inlier)
        elif type1 == "mixhop-contaminated":
            tag_counts = random_split_counts(n, no_of_tags)
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0).generate_graph_contaminated(n, 2, 10, h_inlier)
        elif type1 == "mixhop-disjoint":
            tag_counts_1 = random_split_counts(n//2, no_of_tags)
            g1 = MixhopGraphGenerator(tag_counts_1, heteroWeightsExponent=1.0)(n//2, 2, 10, h_inlier+0.2)
            tag_counts_2 = random_split_counts(n//2, no_of_tags)
            g2 = MixhopGraphGenerator(tag_counts_2, heteroWeightsExponent=1.0)(n//2, 2, 10, h_inlier-0.2)
            g = nx.disjoint_union(g1,g2)
            #tags = [g.nodes[v]['color'] for v in g.nodes]
        
        g = from_networkx(g)
        g.y = torch.tensor([0])

        g_list.append(g)
    
    for i in range(num_test_outlier):
        
        n = np.random.randint(n_min, n_max)
        
        if type2 == "mixhop":
            tag_counts = random_split_counts(n, no_of_tags)
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0)(n, 2, 10, h_outlier)
        elif type2 == "mixhop-contaminated":
            tag_counts = random_split_counts(n, no_of_tags)
            g = MixhopGraphGenerator(tag_counts, heteroWeightsExponent=1.0).generate_graph_contaminated(n, 2, 10, h_outlier)
        elif type2 == "mixhop-disjoint":
            tag_counts_1 = random_split_counts(n//2, no_of_tags)
            g1 = MixhopGraphGenerator(tag_counts_1, heteroWeightsExponent=1.0)(n//2, 2, 10, h_outlier+0.2)
            tag_counts_2 = random_split_counts(n//2, no_of_tags)
            g2 = MixhopGraphGenerator(tag_counts_2, heteroWeightsExponent=1.0)(n//2, 2, 10, h_outlier-0.2)
            g = nx.disjoint_union(g1,g2)
            #tags = [g.nodes[v]['color'] for v in g.nodes]
        
        g = from_networkx(g)
        g.y = torch.tensor([1])

        g_list.append(g)
    
    # Extracting unique tags and converting to one-hot features   
    tagset = set()
    for g in g_list:
        tagset = tagset.union(set(g.color.tolist()))
        
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.x = torch.zeros(len(g.color.tolist()), len(tagset))
        g.x[range(len(g.color.tolist())), [tag2index[tag] for tag in g.color.tolist()]] = 1
        del g.color

    print('Maximum node tag: %d' % len(tagset))

    return SimpleGraphDataset("MIXHOP", g_list)

def mod_CH(X, nu=0.05):
    X = np.sort(X)
    
    outlier_mean = np.mean(X[int((1-nu)*len(X)):])
    for i in range(int((1-nu)*len(X)), len(X)):
        X[i] = outlier_mean
    
    labels = np.array([0]*int((1-nu)*len(X)) + [1]*(len(X) - int((1-nu)*len(X))))
    X = X.reshape(-1, 1)
    from sklearn.metrics import calinski_harabasz_score
    score = calinski_harabasz_score(X, labels)
    return score