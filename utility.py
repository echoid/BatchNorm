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



    return Xtrain, Xtest, Xtrain_mask, Xtest_mask , train_input, test_input , N, D


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
    test_input = train_Mask * trainX + (1 - train_Mask) * 0
    return trainX, testX, train_Mask, test_Mask, train_input, test_input,No ,Dim


class MyDataset(Dataset):
    def __init__(self, X, M, input):
        self.X = torch.tensor(X).float()
        self.M = torch.tensor(M).float()
        self.input = torch.tensor(input).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.input[idx]