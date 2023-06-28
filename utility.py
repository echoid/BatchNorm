import torch
import numpy as np
import torch.nn as nn
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

train_rate = 0.8
p_miss = 0.2


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
    train_input = train_Mask * trainX + (1 - train_Mask) * train_Z

    test_Z = sample_Z(testX.shape[0], Dim)
    test_input = test_Mask * testX + (1 - test_Mask) * test_Z

    return trainX, testX, train_Mask, test_Mask, train_input, test_input,No ,Dim


class MyDataset(Dataset):
    def __init__(self, X, M, input):
        self.X = torch.tensor(X)
        self.M = torch.tensor(M)
        self.input = torch.tensor(input)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.input[idx]