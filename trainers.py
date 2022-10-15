#trainers.py

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_scatter import scatter as sctr


class MMDTrainer:
    
    def __init__(self, model, optimizer, landmark_loader, alpha=1.0, beta=0.0, device=torch.device("cpu"), nystrom="LLSVM", regularizer="variance", kernel_batch=64):
        
        self.debug_mode = False

        self.device = device
        self.nystrom = nystrom

        self.model = model
        self.optimizer = optimizer

        self.center = None
        self.reg_weight = 0
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer

        self.landmark_loader = landmark_loader
        self.gamma = None
        self.kernel_batch = kernel_batch

    def compute_gamma(self, embeddings):
        all_vertex_embeddings = torch.cat(embeddings, axis=0).detach().to(self.device)
        if torch.any(torch.isnan(all_vertex_embeddings)):
            raise ValueError("NaN in embeddings")
        all_vertex_distances = torch.cdist(all_vertex_embeddings, all_vertex_embeddings)**2
        if torch.any(torch.isnan(all_vertex_distances)):
            raise ValueError("NaN in distances")
        median_of_distances = torch.median(all_vertex_distances)
        if median_of_distances <= 1e-4:
            median_of_distances = torch.tensor(1e-4).to(self.device)
        
        gamma = 1/median_of_distances

        return gamma

    def compute_mmd_gram_matrix(self, X_embeddings, Y_embeddings=None, type="SMM"):
        
        if not Y_embeddings:
            Y_embeddings = X_embeddings

        if self.gamma == None:
            self.gamma = self.compute_gamma(Y_embeddings)
        if self.gamma==0:
            raise ValueError("Gamma value appears to be 0")

        gram_matrix = torch.empty(len(X_embeddings), len(Y_embeddings))

        for ix in range(0,len(X_embeddings), self.kernel_batch):
            X_embeddings_batched = X_embeddings[ix:ix+self.kernel_batch]

            for iy in range(0, len(Y_embeddings), self.kernel_batch):
                Y_embeddings_batched = Y_embeddings[iy:iy+self.kernel_batch]

                X_all = torch.cat(X_embeddings_batched).to(self.device)
                Y_all = torch.cat(Y_embeddings_batched).to(self.device)
                
                X_sq = torch.squeeze(torch.matmul(X_all[:,None,:],X_all[:,:,None]))
                XY = torch.matmul(X_all, torch.transpose(Y_all, 0,1))
                del X_all
                
                Y_sq = torch.squeeze(torch.matmul(Y_all[:,None,:],Y_all[:,:,None]))
                del Y_all
                
                Z = torch.exp(-self.gamma * (-2*XY + X_sq[:,None] + Y_sq[None,:]))
                del X_sq, Y_sq, XY

                X_indices = []
                for i, emb in enumerate(X_embeddings_batched):
                    X_indices += [i]*emb.shape[0]

                X_indices = torch.tensor(X_indices).to(self.device)

                temp = sctr(Z, X_indices, dim=0, reduce="mean")
                del Z, X_indices

                Y_indices = []
                for i, emb in enumerate(Y_embeddings_batched):
                    Y_indices += [i]*emb.shape[0]

                Y_indices = torch.tensor(Y_indices).to(self.device)

                gram_matrix[ix:ix+self.kernel_batch, iy:iy+self.kernel_batch] = sctr(temp, Y_indices, dim=1, reduce="mean")
                del Y_indices, temp

        return gram_matrix
    
    
    def train(self, train_loader):
        self.model.train()
        
        if self.center == None:  # first iteration
            F_list = []

        loss_accum = 0
        svdd_loss_accum = 0
        reg_loss_accum = 0
        total_iters = 0

        if self.debug_mode:
            torch.autograd.set_detect_anomaly(True)

        for batch in train_loader:

            landmark_embeddings = []
            for landmark_batch in self.landmark_loader:
                landmark_batch_embeddings = self.model(landmark_batch)
                landmark_embeddings = landmark_embeddings + landmark_batch_embeddings

            self.gamma = self.compute_gamma(landmark_embeddings).detach() # no backpropagation for gamma
            
            train_embeddings = self.model(batch)
            K_trainZ = self.compute_mmd_gram_matrix(train_embeddings, landmark_embeddings).to(self.device)

            if self.nystrom == "LLSVM":
                
                K_Z = self.compute_mmd_gram_matrix(landmark_embeddings).to(self.device)

                K_temp = K_Z.detach()
                eps_matrix = torch.randn_like(K_Z)*torch.median(torch.abs(K_temp))*1e-4
                eigenvalues, U_Z = torch.symeig(K_Z+eps_matrix,eigenvectors=True)

                #removed smallest 2/3 eigenvalues due to numerical instability
                no_of_eigens = len(eigenvalues)
                eigenvalues = eigenvalues[-no_of_eigens//3:]
                
                # if eigenvalues still negative, adjust - values small enough so that it does not affect
                m = min(eigenvalues).detach()
                if m < 0:
                    eigenvalues = eigenvalues - 2*m
                elif m == 0:
                    eigenvalues = eigenvalues + 1e-9
                
                U_Z = U_Z[:,-no_of_eigens//3:]
                T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))
                F_train = torch.matmul(K_trainZ, T)

            elif self.nystrom == "RSVM":
                
                F_train = K_trainZ
                
            

            # if first iteration, compute center, and don't do any backprop
            if self.center == None:
                F_list.append(F_train)
            
            else:
                train_scores = torch.sum((F_train - self.center)**2, dim=1).cpu()
                svdd_loss = torch.mean(train_scores)
                
                #backpropagate
                self.optimizer.zero_grad()
                svdd_loss.backward()
                self.optimizer.step()
                
                svdd_loss_accum += svdd_loss.detach().cpu().numpy()
                total_iters += 1

        if self.center == None:
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() # no backpropagation for center
            #print("center computed")

            average_svdd_loss = -1
        
        else:
            average_svdd_loss = svdd_loss_accum/total_iters

        return average_svdd_loss


    def test(self, test_loader):
        self.model.eval()
        
        with torch.no_grad():

            landmark_embeddings = []

            for landmark_batch in self.landmark_loader:
                landmark_batch_embeddings = self.model(landmark_batch)
                landmark_embeddings = landmark_embeddings + landmark_batch_embeddings

            self.gamma = self.compute_gamma(landmark_embeddings) # no backpropagation for gamma
            
            if self.nystrom == "LLSVM":
                
                K_Z = self.compute_mmd_gram_matrix(landmark_embeddings).to(self.device)
                eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)

                #removed smallest 2/3 eigenvalues due to numerical instability
                no_of_eigens = len(eigenvalues)
                eigenvalues = eigenvalues[-no_of_eigens//3:]
                
                # if eigenvalues still negative, adjust - values small enough so that it does not affect
                m = min(eigenvalues)
                if m < 0:
                    print("neg")
                    eigenvalues = eigenvalues - 2*m
                elif m == 0:
                    print("zero")
                    eigenvalues = eigenvalues + 1e-9
                
                U_Z = U_Z[:,-no_of_eigens//3:]
                T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))
            
            dists_list = []
            for batch in test_loader:

                R_embeddings = self.model(batch)
                K_RZ = self.compute_mmd_gram_matrix(R_embeddings, landmark_embeddings).to(self.device)
                
                if self.nystrom == "LLSVM":
                    F = torch.matmul(K_RZ, T)
                elif self.nystrom == "RSVM":
                    F = K_RZ
                
                batch_dists = torch.sum((F - self.center)**2, dim=1).cpu()
                dists_list.append(batch_dists)
            
            labels = torch.cat([batch.y for batch in test_loader])
            dists = torch.cat(dists_list)
            
            ap = average_precision_score(labels, dists)
            roc_auc = roc_auc_score(labels, dists)

            return ap, roc_auc, dists, labels

class MeanTrainer:
    
    def __init__(self, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
        
        self.device = device

        self.model = model
        self.optimizer = optimizer

        self.center = None
        self.reg_weight = 0
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer        

    def train(self, train_loader):
        self.model.train()
        
        if self.center == None:  # first iteration
            F_list = []

        loss_accum = 0
        svdd_loss_accum = 0
        reg_loss_accum = 0
        total_iters = 0

        for batch in train_loader:
            
            train_embeddings = self.model(batch)
            mean_train_embeddings = [torch.mean(emb, dim=0) for emb in train_embeddings] # Mean-ggregation: G_emb = mean(v_emb for v in G)
            F_train = torch.stack(mean_train_embeddings)
                
            # if first iteration, compute center, and don't do any backprop
            if self.center == None:
                F_list.append(F_train)
            
            else:
                train_scores = torch.sum((F_train - self.center)**2, dim=1).cpu()
                svdd_loss = torch.mean(train_scores)
                
                #backpropagate
                self.optimizer.zero_grad()
                svdd_loss.backward()    
                self.optimizer.step()
                
                svdd_loss_accum += svdd_loss.detach().cpu().numpy()
                total_iters += 1

        if self.center == None: # first epoch only
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() # no backpropagation for center

            average_svdd_loss = -1
        else:
            average_svdd_loss = svdd_loss_accum/total_iters

        return average_svdd_loss


    def test(self, test_loader):
        self.model.eval()
        
        with torch.no_grad():

            dists_list = []
            for batch in test_loader:

                test_embeddings = self.model(batch)
                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings] # Mean-aggregation: G_emb = mean(v_emb for v in G)
                F_test = torch.stack(mean_test_embeddings)
                
                batch_dists = torch.sum((F_test - self.center)**2, dim=1).cpu()
                dists_list.append(batch_dists)
            
            labels = torch.cat([batch.y for batch in test_loader])
            dists = torch.cat(dists_list)

            ap = average_precision_score(labels, dists)
            roc_auc = roc_auc_score(labels, dists)

            return ap, roc_auc, dists, labels