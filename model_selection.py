# model_selection.py

import os
import argparse
import numpy as np
import pickle
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score, average_precision_score, roc_auc_score


def reliability_scores(D, narrow=False, measure="spearman", aggregation="mean", preprocess="none", idx='svdd', non_seed_indices=[0]):
    keys = D.keys()
    total_models = len(keys)

    if narrow:
        distinct_models = list(set([tuple(k[i] for i in non_seed_indices) for k in keys]))
        #distinct_models = list(set([(k[0], k[1], k[-1]) for k in keys]))
        runs_per_model = total_models // len(distinct_models)
        print("runs per model: ", runs_per_model)
        similarity_matrix = np.zeros((total_models, runs_per_model-1))
    else:
        similarity_matrix = np.zeros((total_models, total_models-1))

    for i,k in enumerate(keys):
        kk = tuple(k[i] for i in non_seed_indices)
        try:
            dists_k = D[k][idx].dists.numpy()
        except:
            dists_k = D[k][idx].dists

        if preprocess=="rank":
            temp = (-dists_k).argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(1,len(dists_k)+1)
            dists_k = 1/ranks
        
        if narrow:
            other_runs = [j for j in keys if j != k and tuple(j[i] for i in non_seed_indices) == kk]
        
        else:
            other_runs = [j for j in keys if j != k]
        
        for j,l in enumerate(other_runs):
            try:
                dists_l = D[l][idx].dists.numpy()
            except:
                dists_l = D[l][idx].dists

            if preprocess=="maxnormalize":
                dists_l  = dists_l/maxdist
            elif preprocess=="minmaxnormalize":
                dists_l = (dists_l - mindist)/(maxdist-mindist)
            elif preprocess=="rank":
                temp = (-dists_l).argsort()
                ranks = np.empty_like(temp)
                ranks[temp] = np.arange(1,len(dists_l)+1)
                dists_l = 1/ranks
        
            if measure == "spearman":
                score = spearmanr(dists_k, dists_l)[0]
            elif measure == "KT":
                score = kendalltau(dists_k, dists_l)[0]
            elif measure == "NDCG":
                score = (ndcg_score(np.asarray([dists_k]), np.asarray([dists_l])) + ndcg_score(np.asarray([dists_k]), np.asarray([dists_l])))/2
            else:
                print("wrong measure")
                return -1
            
            similarity_matrix[i,j] = score
            
    if aggregation == "mean":
        reliability_scores = np.mean(similarity_matrix, 1)
    elif aggregation == "median":
        reliability_scores = np.median(similarity_matrix, 1)
    else:
        print("wrong aggregation")
        return -1

    return reliability_scores




def HITS(D, init = "rank", idx='svdd'):
    keys = D.keys()
    os_lists = [D[k][idx].dists for k in keys]
    no_of_models = len(os_lists)
    no_of_points = len(os_lists[0])
    
    if init == "scores":
        label_list = os_lists
    elif init == "rank":
        label_list = []
        for os_list in os_lists:
            temp = (-np.array(os_list)).argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(1,len(os_list)+1)
            labels = (1/ranks).tolist()
            label_list.append(labels)

    model_weights = np.ones(no_of_models)


    while True:
        model_weights_old = np.copy(model_weights)

        label_matrix = np.stack(label_list)
        point_weights = np.matmul(model_weights, label_matrix)
        point_weights = point_weights/np.linalg.norm(point_weights)
        
        model_weights = np.matmul(label_matrix, point_weights)
        model_weights = model_weights/np.linalg.norm(model_weights)

        best_model = np.argmax(model_weights)

        if np.max(np.abs(model_weights_old - model_weights)) < 1e-4:
            break


    return list(keys)[best_model], point_weights


def compute_model_selection(filenames, non_seed_indices, idx_list):

    D = {}
    
    for file_idx,filename in enumerate(filenames):
        
        with open(filename, 'rb') as f:
            D1 = pickle.load(f)
        
        for k,v in D1.items():
            D[k+(file_idx,)] = v
    

    non_seed_indices.append(len(list(D.keys())[0])-1)

    for idx in idx_list:
        
        if idx == 'svdd':
            print("Epoch selection: min SVDD")
        elif idx == 'last':
            print("Epoch selection: 150th epoch")
        elif idx == 'default':
            print("Running for two-stage, no epoch selection")

        all_aps = []
        all_roc_aucs = []
        
        if idx != 'default':
              min_svdd_model = None
              min_svdd = np.inf
        
        for k in D.keys():
            model = D[k][idx]
            all_aps.append(model.ap)
            all_roc_aucs.append(model.roc_auc)

            if idx != 'default':
                if model.svdd_loss < min_svdd:
                    min_svdd_model = model
                    min_svdd = model.svdd_loss

        print("\tAverage: AP=%.2f +- %.2f, ROC-AUC=%.2f += %.2f" % (np.mean(all_aps), np.std(all_aps), np.mean(all_roc_aucs), np.std(all_roc_aucs)))
        if idx != 'default':
            print("\tAt Min SVDD: AP=%.2f, ROC-AUC=%.2f" % (min_svdd_model.ap, min_svdd_model.roc_auc))
        
        
        rel = reliability_scores(D, narrow=False, measure="spearman", aggregation="median", preprocess="none", idx=idx, non_seed_indices=non_seed_indices)
        max_idx = np.argmax(rel)
        v = list(D.values())[max_idx]
        print("\tMC: AP=%.2f, ROC-AUC=%.2f" % (v[idx].ap, v[idx].roc_auc))

        if len(filenames)==1:
            rel = reliability_scores(D, narrow=True, measure="spearman", aggregation="mean", preprocess="none", idx=idx, non_seed_indices=non_seed_indices)
            max_idx = np.argmax(rel)
            max_udr = np.max(rel)
            print("max UDR:", max_udr)
            v = list(D.values())[max_idx]
            print("\tUDR: AP=%.2f, ROC-AUC=%.2f" % (v[idx].ap, v[idx].roc_auc))
        

        for init in ["scores", "rank"]:
            
            if init=="scores":
                print("\tHITS using actual scores:")
            elif init=="rank":
                print("\tHITS using 1/rank:")
            best_hits_key, combined_hits_scores = HITS(D, init=init, idx=idx)
            print("\t\tBest model: AP=%.2f, ROC-AUC=%.2f" % (D[best_hits_key][idx].ap, D[best_hits_key][idx].roc_auc))

            labels = D[best_hits_key][idx].labels
            ap = average_precision_score(labels, combined_hits_scores)
            roc_auc = roc_auc_score(labels, combined_hits_scores)
            print("\t\tEnsemble model: AP=%.2f, ROC-AUC=%.2f" % (ap, roc_auc))

        print("\n\n")


parser = argparse.ArgumentParser(description='Model Selection')

parser.add_argument('--data', default='mixhop',
                    help='dataset name (default: mixhop)')
parser.add_argument('--data_seed', type=int, default=1213,
                    help='seed to split the inlier set into train and test (default: 1213)')
parser.add_argument('--aggregation', type=str, default="both", choices=["MMD", "Mean", "both"],
                    help='Type of graph level aggregation (default: both)')

args = parser.parse_args()

if args.aggregation == "both":
    filenames = ['outputs/GIN_MMD_models_' + args.data + '_' + str(args.data_seed) + '.pkl', 'outputs/GIN_Mean_models_' + args.data + '_' + str(args.data_seed) + '.pkl']
else:
    filenames = ['outputs/GIN_'+ args.aggregation + '_models_' + args.data + '_' + str(args.data_seed) + '.pkl']

compute_model_selection(filenames=filenames, idx_list=['svdd'], non_seed_indices=[0,1,4,5])
