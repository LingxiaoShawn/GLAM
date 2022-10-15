#dataloader.py

import torch, os
import numpy as np
import random
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.utils import degree
from pathlib import Path
import math
import pickle

from utils import load_synthetic_data

DATA_PATH = 'downloads'
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

class RemoveLastKFeatures(object):
    r"""Removes the last k features.

    Args:
        k (int)
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        
        if self.k == 0:
            return data
        elif self.k < 0:
            raise ValueError("k cannot be negative.")
        else:
            x = data.x

            if x is not None:
                x = x.view(-1, 1) if x.dim() == 1 else x
                if x.shape[1]>self.k:
                    c = x[:,:-self.k]
                    data.x = c
                else:
                    data.x = None
            else:
                data.x = None

            return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class DownsamplingFilter(object):
    def __init__(self, min_nodes, max_nodes, down_class, down_rate, num_classes, reverse=True, coin=np.random.default_rng()):
        super(DownsamplingFilter, self).__init__()
        # if not reverse, downsampling mentioned class, and mentioned class as anomaly class
        # if reverse, downsampling unmentioned class, and mentioned class as normal class
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.down_class = down_class
        self.down_rate = down_rate
        self.reverse = reverse
        self.coin = coin
        

    def __call__(self, data):
        # step 1: filter the graph node size
        keep = (data.num_nodes <= self.max_nodes) and (data.num_nodes >= self.min_nodes)
        # for graph classification, down_rate is 1
        # downsampling only for anomaly detection, not for classification
        if self.down_rate == 1:
            return keep
        if keep:
            # step 2: downsampling class
            mentioned_class = (data.y.item() == self.down_class)
            
            anomalous_class = not mentioned_class if self.reverse else mentioned_class
            data.y.fill_(int(anomalous_class)) # anomalous class as positive

            if anomalous_class:
                if self.coin.random() > self.down_rate:
                    keep = False
        return keep

def load_data(data_name, down_class=0, down_rate=1, use_node_labels=True, use_node_attr=False, dense=False, ignore_edge_weight=True, seed=1213, save_indices_to_disk=True):
    
    np.random.seed(seed)
    newcoin = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    if os.path.exists(DATA_PATH + "/" + data_name + "_" + str(seed) + ".pkl"):
        with open(DATA_PATH + "/" + data_name + "_" + str(seed) + ".pkl", 'rb') as f:
            dataset_raw = pickle.load(f)
    elif data_name == 'mixhop':
        NUM = 500
        dataset_raw = load_synthetic_data(num_train=NUM, num_test_inlier=NUM, num_test_outlier=int(NUM*down_rate), seed=seed)
        with open(DATA_PATH + "/" + data_name + "_" + str(seed) + ".pkl", 'wb') as f:
            pickle.dump(dataset_raw, f)
    else:
        if use_node_labels:
            dataset_raw = TUDataset(root=DATA_PATH, name=data_name, use_node_attr=use_node_attr)
        else:
            temp = TUDataset(root=DATA_PATH, name=data_name, use_node_attr=False)
            no_of_labels = temp.data.x.shape[1]
            dataset_raw = TUDataset(root=DATA_PATH, name=data_name, transform=RemoveLastKFeatures(no_of_labels), use_node_attr=use_node_attr)

    if len(dataset_raw) > 10000:
        retaincoin = np.random.default_rng(seed)
        retained = retaincoin.choice(len(dataset_raw), 10000, replace=False)
        dataset_raw = dataset_raw[torch.tensor(retained)]
    
    # downsampling 
    num_nodes_graphs = [data.num_nodes for data in dataset_raw]
    min_nodes, max_nodes = min(num_nodes_graphs), max(num_nodes_graphs)
    if max_nodes >= 10000:
        max_nodes = 10000


    if use_node_labels and (not use_node_attr):
    	INDICES_PATH = "data/labeled"
    elif (not use_node_labels) and use_node_attr:
    	INDICES_PATH = "data/attributed"
    else:
    	raise ValueError("Exactly ONE of use_node_attr and use_node_labels must be True")

    if not os.path.isdir(INDICES_PATH):
	    os.mkdir(INDICES_PATH)

    if os.path.exists(INDICES_PATH + "/" + data_name + "_" + str(seed) + "(in" + str(down_class) + ")" + "_INDICES.pkl"):
        print("Using saved indices to extract TRAIN and TEST datasets...")
        save_indices_to_disk = False
        with open(INDICES_PATH + "/" + data_name + "_" + str(seed) + "(in" + str(down_class) + ")" + "_INDICES.pkl", 'rb') as f:
            downsampled_indices, train_indices = pickle.load(f)
    else:
        print("Downsampling and splitting into TRAIN and TEST datasets using seed ", seed, "...")

        inliers = [data for data in dataset_raw if data.y == down_class]

        Ni = len(inliers)
        No = len(dataset_raw) - Ni
        down_rate = 0.5*down_rate*Ni/No

        filter = DownsamplingFilter(min_nodes, max_nodes, down_class, down_rate, dataset_raw.num_classes, reverse=True, coin=newcoin)
        downsampled_indices = [i for i, data in enumerate(dataset_raw) if filter(data)]
    

    dataset = dataset_raw[torch.tensor(downsampled_indices)]
    
    # preprocessing: do not use original edge features or weights
    if ignore_edge_weight:
        dataset.data.edge_attr = None

    # add transforms which will be conducted when draw each elements from the dataset
    if dataset.data.x is None:
        print('Using degrees as labels...')
        max_degree = 0
        degs = []
        for data in dataset_raw: # ATTENTION: use dataset_raw instead of downsampled version!
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
            # dataset.num_features = max_degree
        else:
            # dataset['num_features'] = 1
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    if dense:
        if dataset.transform is None:
            dataset.transform = T.ToDense(max_nodes)
        else:
            dataset.transform = T.Compose([dataset.transform, T.ToDense(max_nodes)])


    dataset_list = [data for data in dataset]

    if not os.path.exists(INDICES_PATH + "/" + data_name + "_" + str(seed) + "(in" + str(down_class) + ")" + "_INDICES.pkl"):
        train_indices = [i for i, data in enumerate(dataset_list) if data.y.item()==0 and newcoin.random()<0.5]
        
    train_dataset = [dataset_list[idx] for idx in train_indices] # only keep normal class left
    test_dataset = [dataset_list[idx] for idx in range(len(dataset_list)) if idx not in train_indices]
    

    if save_indices_to_disk:
        with open(INDICES_PATH + "/" + data_name + "_" + str(seed) + "(in" + str(down_class) + ")" + "_INDICES.pkl", 'wb') as f:
            pickle.dump((downsampled_indices, train_indices), f)

    return train_dataset, test_dataset


def create_loaders(data_name, batch_size=64, down_class=0, down_rate=0.05, use_node_attr=False, use_node_labels=True, dense=False, classify=False, one_class_train=True, data_seed=1213, landmark_seed=0, landmark_set_size=4, save_indices_to_disk=True):

    train_dataset, test_dataset = load_data(data_name, 
                                            down_class=down_class, 
                                            down_rate=down_rate,
                                            use_node_attr=use_node_attr, 
                                            use_node_labels=use_node_labels, 
                                            dense=dense, 
                                            seed=data_seed,
                                            save_indices_to_disk=save_indices_to_disk)

    k = int(landmark_set_size*math.log2(len(train_dataset)))
    random.seed(landmark_seed)
    landmark_set = random.sample(train_dataset, k)

    #print("After downsampling and test-train splitting, distribution of classes:")
    labels = np.array([data.y.item() for data in train_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TRAIN: Number of graphs: %d, Class distribution %s"%(len(train_dataset), label_dist))
    
    labels = np.array([data.y.item() for data in test_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("TEST: Number of graphs: %d, Class distribution %s"%(len(test_dataset), label_dist))
    print("Number of node features: %d" %(train_dataset[0].num_features))

    Loader = DenseDataLoader if dense else DataLoader
    num_workers = 0
    train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = Loader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)
    landmark_loader = Loader(landmark_set, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader, landmark_loader, train_dataset[0].num_features