import torch
import numpy as np
import torch.nn as nn
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle

train_rate = 0.8
p_miss = 0.2

p_hint = 0.9

def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B
    return C
def check_and_fill_nan(array, reference_array):
    num_rows, num_cols = array.shape

    for col_idx in range(num_cols):
        if np.all(np.isnan(array[:, col_idx])):
            array[0, col_idx] = np.nanmean(reference_array[:, col_idx])

    return array

def load_dataloader(dataname,missing_type = "quantile", missing_name = "Q4_complete",seed = 1):

    processed_data_path_norm = (
            f"../MNAR/datasets/{dataname}/{missing_type}-{missing_name}_seed-{seed}_max-min_norm.pk"
        )
    with open(processed_data_path_norm, "rb") as f:
            observed_values, observed_masks, gt_masks, eval_length = pickle.load(
                    f
            )

    N, D = observed_values.shape

    
    indlist = np.arange(N)

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / 5
    start = (int)((5 - 1) * N * tmp_ratio)
    
    end = (int)(5 * N * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)

    # Modify here to change train,valid ratio
    num_train = (int)(len(remain_index) * 0.9)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]


    Xtrain = observed_values[train_index]
    Xtest = observed_values[test_index]
    Xval_org = observed_values[valid_index]

    Xtrain_mask = gt_masks[train_index]
    Xtest_mask = gt_masks[test_index]
    Xval_org_mask = gt_masks[valid_index]

    train_Z = sample_Z(Xtrain.shape[0], D)
    test_Z = sample_Z(Xtest.shape[0], D)

    train_input = Xtrain_mask * Xtrain + (1 - Xtrain_mask) * train_Z

    test_input = Xtest_mask * Xtest + (1 - Xtest_mask) * test_Z

    # train_input = Xtrain_mask * Xtrain + (1 - Xtrain_mask) * 0

    # test_input = Xtest_mask * Xtest + (1 - Xtest_mask) * 0


    train_H = sample_M(Xtrain.shape[0], D, 1-p_hint)
    train_H = Xtrain_mask * train_H


    test_H = sample_M(Xtest.shape[0], D, 1-p_hint)
    test_H = Xtest_mask * test_H


    return Xtrain, Xtest, Xtrain_mask, Xtest_mask , train_input, test_input , N, D, train_H, test_H


def load_dataloader(dataname,missing_type = "quantile", missing_name = "Q4_complete",seed = 1):

    processed_data_path_norm = (
            f"../MNAR/datasets/{dataname}/{missing_type}-{missing_name}_seed-{seed}_max-min_norm.pk"
        )
    with open(processed_data_path_norm, "rb") as f:
            observed_values, observed_masks, gt_masks, eval_length = pickle.load(
                    f
            )

    N, D = observed_values.shape

    
    indlist = np.arange(N)

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / 5
    start = (int)((5 - 1) * N * tmp_ratio)
    
    end = (int)(5 * N * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)

    # Modify here to change train,valid ratio
    num_train = (int)(len(remain_index) * 0.9)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]


    Xtrain = observed_values[train_index]
    Xtest = observed_values[test_index]
    Xval_org = observed_values[valid_index]

    Xtrain_mask = gt_masks[train_index]
    Xtest_mask = gt_masks[test_index]
    Xval_org_mask = gt_masks[valid_index]

    train_Z = sample_Z(Xtrain.shape[0], D)
    test_Z = sample_Z(Xtest.shape[0], D)
    val_Z = sample_Z(Xval_org.shape[0], D)

    train_input = Xtrain_mask * Xtrain + (1 - Xtrain_mask) * train_Z

    test_input = Xtest_mask * Xtest + (1 - Xtest_mask) * test_Z

    val_input = Xval_org_mask * Xval_org + (1 - Xval_org_mask) * val_Z


    train_input = Xtrain_mask * Xtrain + (1 - Xtrain_mask) * 0

    test_input = Xtest_mask * Xtest + (1 - Xtest_mask) * 0

    val_input = Xval_org_mask * Xval_org + (1 - Xval_org_mask) * 0


    train_H = sample_M(Xtrain.shape[0], D, 1-p_hint)
    train_H = Xtrain_mask * train_H


    test_H = sample_M(Xtest.shape[0], D, 1-p_hint)
    test_H = Xtest_mask * test_H


    val_H = sample_M(Xval_org.shape[0], D, 1-p_hint)
    val_H = Xval_org_mask * val_H


    return Xtrain, Xtest, Xtrain_mask, Xtest_mask , train_input, test_input , N, D, train_H, test_H, Xval_org, Xval_org_mask, val_input, val_H


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)
    


# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n]) 


def preprocess(dataset_file,train_rate = 0.8,p_miss = 0.2):

    # Data generation
    Data = np.loadtxt(dataset_file, delimiter=",",skiprows=1)

    # Parameters
    No = len(Data)
    Dim = len(Data[0,:])

    # Hidden state dimensions
    H_Dim1 = Dim
    H_Dim2 = Dim

    # Normalization (0 to 1)
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    for i in range(Dim):
        Min_Val[i] = np.min(Data[:,i])
        Data[:,i] = Data[:,i] - np.min(Data[:,i])
        Max_Val[i] = np.max(Data[:,i])
        Data[:,i] = Data[:,i] / (np.max(Data[:,i]) + 1e-6)    

    #%% Missing introducing
    p_miss_vec = p_miss * np.ones((Dim,1)) 
    
    Missing = np.zeros((No,Dim))

    for i in range(Dim):
        A = np.random.uniform(0., 1., size = [len(Data),])
        B = A > p_miss_vec[i]
        Missing[:,i] = 1.*B

        
    #%% Train Test Division    
    
    idx = np.random.permutation(No)

    Train_No = int(No * train_rate)
    Test_No = No - Train_No

    # Train / Test / Validation Features
    trainX = Data[:Train_No, :]
    testX = Data[Train_No:, :]

    # Train / Test / Validation Missing Mask 0=missing, 1=observed
    train_Mask = Missing[:Train_No, :]
    test_Mask = Missing[Train_No:, :]


    train_Z = sample_Z(trainX.shape[0], Dim)
    #train_input = train_Mask * trainX + (1 - train_Mask) * train_Z
    train_input = train_Mask * trainX + (1 - train_Mask) * 0

    test_Z = sample_Z(testX.shape[0], Dim)
    #test_input = test_Mask * testX + (1 - test_Mask) * test_Z
    test_input = test_Mask * testX + (1 - test_Mask) * 0
    return trainX, testX, train_Mask, test_Mask, train_input, test_input,No ,Dim


class MyDataset(Dataset):
    def __init__(self, X, M, input,h):
        self.X = torch.tensor(X).float()
        self.M = torch.tensor(M).float()
        self.input = torch.tensor(input).float()
        self.h = torch.tensor(h).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.input[idx],self.h[idx]
    


class Imputation_model(nn.Module):
    def __init__(self, dim, hidden_dim1, hidden_dim2,use_BN):
        super(Imputation_model, self).__init__()
    
        self.G_W1 = nn.Parameter(torch.tensor(xavier_init([dim * 2, hidden_dim1]), dtype=torch.float32), requires_grad=True)
        self.G_b1 = nn.Parameter(torch.zeros(hidden_dim1, dtype=torch.float32), requires_grad=True)
        self.G_bn1 = nn.BatchNorm1d(hidden_dim1)

        self.G_W2 = nn.Parameter(torch.tensor(xavier_init([hidden_dim1, hidden_dim2]), dtype=torch.float32), requires_grad=True)
        self.G_b2 = nn.Parameter(torch.zeros(hidden_dim2, dtype=torch.float32), requires_grad=True)
        self.G_bn2 = nn.BatchNorm1d(hidden_dim2)

        self.G_W3 = nn.Parameter(torch.tensor(xavier_init([hidden_dim2, dim]), dtype=torch.float32), requires_grad=True)
        self.G_b3 = nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)

        self.use_BN = use_BN
        self.batch_mean1 = None
        self.batch_var1 = None

        self.batch_mean2 = None
        self.batch_var2 = None

    def forward(self, data, mask):
        inputs = torch.cat(dim=1, tensors=[data, mask])  # Mask + Data Concatenate
        inputs = inputs.float()  
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1.float()) + self.G_b1.float())
        if self.use_BN:
            G_h1 = self.G_bn1(G_h1)  # Batch Normalization
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2.float()) + self.G_b2.float())
        if self.use_BN:
            G_h2 = self.G_bn2(G_h2)  # Batch Normalization
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3.float()) + self.G_b3.float())  # [0,1] normalized Output

        if self.use_BN:
            self.batch_mean1 = self.G_bn1.running_mean
            self.batch_var1 = self.G_bn1.running_var
            self.batch_mean2 = self.G_bn2.running_mean
            self.batch_var2 = self.G_bn2.running_var

        return G_prob

def set_all_BN_layers_tracking_state(model, state):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = state

def get_dataset_loaders(trainX, train_Mask,train_input,testX, test_Mask,test_input,train_H, test_H,valX, val_Mask,val_input,val_H):

    train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input,train_H), MyDataset(testX, test_Mask, test_input,test_H)
    val_dataset = MyDataset(valX, val_Mask,val_input,val_H)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader , test_loader, val_loader

def loss(truth, mask, data,imputer):

    generated = imputer(data, mask)

    return  torch.mean(((1 - mask) * truth - (1 - mask) * generated) ** 2) / torch.mean(1 - mask), generated

def impute_with_prediction(original_data, mask, prediction):

    return mask * original_data + (1 - mask) * prediction



# Training loop with early stopping
def train_with_early_stopping(imputer, train_loader, test_loader, epoch, patience):
    # Define early stopping parameters
    best_validation_loss = float('inf')  # Initialize with a high value
    best_model_state = None
    early_stopping_counter = 0

    optimizer = torch.optim.Adam(params=imputer.parameters())

    for it in tqdm(range(epoch)):
        imputer.train()
        total_loss = 0
        batch_no = 0

        for truth_X, mask, data_X, x_hat in train_loader:
            batch_no += 1

            optimizer.zero_grad()

            Imputer_loss = loss(truth=truth_X, mask=mask, data=data_X, imputer=imputer)[0]
            total_loss += Imputer_loss.item()
            Imputer_loss.backward()
            optimizer.step()

        # Calculate average training loss for the epoch
        avg_train_loss = np.sqrt(total_loss / batch_no)

        # Validation step
        imputer.eval()
        with torch.no_grad():
            total_val_loss = 0
            val_batch_no = 0

            for truth_X_val, mask_val, data_X_val, x_hat_val in test_loader:
                val_batch_no += 1

                val_loss = loss(truth=truth_X_val, mask=mask_val, data=data_X_val, imputer=imputer)[0]
                total_val_loss += val_loss.item()

        avg_val_loss = np.sqrt(total_val_loss / val_batch_no)

        # Check for improvement in validation loss
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            best_model_state = imputer.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Print current epoch's training and validation loss
        print('Epoch: {}'.format(it), end='\t')
        print('Train_loss: {:.4}'.format(avg_train_loss), end='\t')
        print('Val_loss: {:.4}'.format(avg_val_loss), end='\n')

        # Check for early stopping
        if early_stopping_counter >= patience:
            print("Early stopping! No improvement in validation loss for {} epochs.".format(patience))
            break

    # Load the best model state after training loop completes
    imputer.load_state_dict(best_model_state)