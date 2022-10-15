#main.py

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import pickle
import argparse
from types import SimpleNamespace

from matplotlib import rcParams
rcParams.update({'figure.autolayout': False})

from dataloader import create_loaders
from GIN import GIN
from trainers import MMDTrainer, MeanTrainer

def run_experiment(
    data = "saved", data_seed=1213, inlier_cls=0, down_rate=0.05, use_node_attr=False, use_node_labels=True,
    alpha=1.0, beta=0.0, epochs=150, model_seed=0, landmark_seed=100, num_layers=1, landmark_set_size=4,
    device=0, aggregation="MMD", nystrom="LLSVM", bias=False, hidden_dim=64, lr=0.1, weight_decay=1e-5, batch = 64, kernel_batch=64
    ):

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # load data
    train_loader, test_loader, landmark_loader, num_features = create_loaders(data_name=data, 
                            batch_size=batch, 
                            down_class=inlier_cls, 
                            down_rate=down_rate,
                            use_node_attr=use_node_attr,
                            use_node_labels=use_node_labels,
                            dense=False,
                            data_seed=data_seed,
                            landmark_seed=landmark_seed,
                            landmark_set_size=landmark_set_size)

    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    model = GIN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if aggregation=="MMD":

        trainer = MMDTrainer(
            model=model,
            optimizer=optimizer,
            landmark_loader=landmark_loader,
            alpha=alpha,
            beta=beta,
            device=device,
            nystrom=nystrom,
            kernel_batch=kernel_batch
            )
    
    elif aggregation=="Mean":
        trainer = MeanTrainer(
            model=model,
            optimizer=optimizer,
            alpha=alpha,
            beta=beta,
            device=device
            )
    
    epochinfo = []

    for epoch in range(epochs+1):

        print("Epoch %3d" % (epoch), end="\t")
        svdd_loss = trainer.train(train_loader=train_loader)
        print("SVDD loss: %f" % (svdd_loss), end="\t")
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)
        #print("AP: %f" % ap, end="\t")
        print("ROC-AUC: %f" % roc_auc)

        
        TEMP = SimpleNamespace()
        TEMP.epoch_no = epoch
        TEMP.dists = dists
        TEMP.labels = labels
        TEMP.ap = ap
        TEMP.roc_auc = roc_auc
        TEMP.svdd_loss = svdd_loss

        epochinfo.append(TEMP)  



    best_svdd_idx = np.argmin([e.svdd_loss for e in epochinfo[1:]])+1
    
    print("      Min SVDD, at epoch %d, AP: %.2f, ROC-AUC: %.2f" % (best_svdd_idx, epochinfo[best_svdd_idx].ap, epochinfo[best_svdd_idx].roc_auc))
    print("    At the end, at epoch %d, AP: %.2f, ROC-AUC: %.2f" % (args.epochs, epochinfo[-1].ap, epochinfo[-1].roc_auc))

    important_epoch_info = {}
    important_epoch_info['svdd'] = epochinfo[best_svdd_idx]
    important_epoch_info['last'] = epochinfo[-1]
    
    return important_epoch_info


parser = argparse.ArgumentParser(description='GLAM: PyTorch graph convolutional neural net for whole-graph anomaly detection')

parser.add_argument('--data', default='mixhop',
                    help='dataset name (default: mixhop)')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size (default: 64)')
parser.add_argument('--data_seed', type=int, default=1213,
                    help='seed to split the inlier set into train and test (default: 1213)')
parser.add_argument('--inlier_cls', type=int, default=0,
                    help='inlier class (default: 0)')
parser.add_argument('--down_rate', type=float, default=0.05,
                    help='outlier/inlier fraction (default: 0.05)')
parser.add_argument('--use_node_attr', action="store_true",
                                    help='Whether to use continuous node attributes (if available).')
parser.add_argument('--ignore_node_labels', action="store_true",
                                    help='Whether to ignore node labels (if available).')


parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')

parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train (default: 150)')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='number of hidden units (default: 64)')
parser.add_argument('--layers', type=int, default=2,
                    help='number of hidden layers (default: 2)')
parser.add_argument('--bias', action="store_true",
                                    help='Whether to use bias terms in the GNN.')

parser.add_argument('--aggregation', type=str, default="MMD", choices=["MMD", "Mean"],
                    help='Type of graph level aggregation (default: MMD)')

parser.add_argument('--use_config', action="store_true",
                                    help='Whether to use configuration from a file')
parser.add_argument('--config_file', type=str, default="configs/config.txt",
                    help='Name of configuration file (default: configs/config.txt)')


parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight_decay constant lambda (default: 1e-4)')
parser.add_argument('--landmark_seed', type=int, default=666, 
                    help='Landmark seed (default: 666)')
parser.add_argument('--model_seed', type=int, default=0, 
                    help='Model seed (default: 0)')
parser.add_argument('--landmark_set_size', type=int, default=4, 
                    help='Landmark set size (default: 4)')

args = parser.parse_args()

lrs = [args.lr]
weight_decays = [args.weight_decay]
layercounts = [args.layers]
landmark_seeds = [args.landmark_seed]
model_seeds = [args.model_seed]
landmark_set_sizes = [args.landmark_set_size]

if args.use_config:

    with open(args.config_file) as f:
        lines = [line.rstrip() for line in f]

    for line in lines:
        words = line.split()
        if words[0] == "LR":
            lrs = [float(w) for w in words[1:]]
        elif words[0] == "WD":
            weight_decays = [float(w) for w in words[1:]]
        elif words[0] == "layers":
            layercounts = [int(w) for w in words[1:]]
        elif words[0] == "landmark_seeds":
            landmark_seeds = [int(w) for w in words[1:]]
        elif words[0] == "model_seeds":
            model_seeds = [int(w) for w in words[1:]]
        elif words[0] == "landmark_set_sizes":
            landmark_set_sizes = [int(w) for w in words[1:]]
        else:
            print("Cannot parse line: ", line)


D = {}

for lr in lrs:
    for weight_decay in weight_decays:
        for landmark_seed in landmark_seeds:
            for model_seed in model_seeds:
                for layercount in layercounts:
                    for landmark_set_size in landmark_set_sizes:
            
                        print("Running experiment for LR=%f, weight decay = %.1E, landmark seed = %d, model seed = %d, number of layers = %d, landmark set size = %d log N" % (lr, weight_decay, landmark_seed, model_seed, layercount, landmark_set_size))
                        D[(lr,weight_decay,landmark_seed,model_seed, layercount, landmark_set_size)] = run_experiment(
                            data=args.data,
                            data_seed=args.data_seed,
                            inlier_cls=args.inlier_cls,
                            down_rate=args.down_rate,
                            use_node_attr=args.use_node_attr,
                            use_node_labels=(not args.ignore_node_labels),
                            epochs=args.epochs,
                            model_seed=model_seed, # SEED
                            landmark_seed=landmark_seed, # SEED
                            num_layers=layercount, # HYPERPARAMETER
                            landmark_set_size=landmark_set_size, # HYPERPARAMETER
                            device=args.device,
                            aggregation=args.aggregation,
                            bias=args.bias,
                            hidden_dim=args.hidden_dim,
                            lr=lr,  # HYPERPARAMETER
                            weight_decay=weight_decay,  # HYPERPARAMETER
                            batch=args.batch
                        )
                        
if args.use_config:
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    with open('outputs/GIN_'+ args.aggregation + '_models_' + args.data + '_' + str(args.data_seed) + '.pkl', 'wb') as f:
        pickle.dump(D, f)
