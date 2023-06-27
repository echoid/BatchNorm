#%% Packages
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utility import xavier_init,MyDataset,sample_Z


dataset_file = 'Spam.csv'  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset
use_gpu = False  # set it to True to use GPU and False to use CPU

if use_gpu:
    torch.cuda.set_device(0)
#%% System Parameters
batch_size = 128
p_miss = 0.2
epoch = 200
train_rate = 0.8

#%% Data

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


# Dataloader
    
train_dataset,test_dataset = MyDataset(trainX, train_Mask), MyDataset(testX, test_Mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    


class Generator(nn.Module):
    def __init__(self, dim, hidden_dim1, hidden_dim2):
        super(Generator, self).__init__()

        self.No = No
        self.Dim = Dim

        self.G_W1 = nn.Parameter(torch.tensor(xavier_init([dim * 2, hidden_dim1]), dtype=torch.float32), requires_grad=True)
        self.G_b1 = nn.Parameter(torch.zeros(hidden_dim1, dtype=torch.float32), requires_grad=True)
        self.G_bn1 = nn.BatchNorm1d(hidden_dim1)

        self.G_W2 = nn.Parameter(torch.tensor(xavier_init([hidden_dim1, hidden_dim2]), dtype=torch.float32), requires_grad=True)
        self.G_b2 = nn.Parameter(torch.zeros(hidden_dim2, dtype=torch.float32), requires_grad=True)
        self.G_bn2 = nn.BatchNorm1d(hidden_dim2)

        self.G_W3 = nn.Parameter(torch.tensor(xavier_init([hidden_dim2, dim]), dtype=torch.float32), requires_grad=True)
        self.G_b3 = nn.Parameter(torch.zeros(dim, dtype=torch.float32), requires_grad=True)


        self.batch_mean1 = None
        self.batch_var1 = None

        self.batch_mean2 = None
        self.batch_var2 = None

    def forward(self, new_x, m):
        inputs = torch.cat(dim=1, tensors=[new_x, m])  # Mask + Data Concatenate
        inputs = inputs.float()  
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1.float()) + self.G_b1.float())
        G_h1 = self.G_bn1(G_h1)  # Batch Normalization
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2.float()) + self.G_b2.float())
        G_h2 = self.G_bn2(G_h2)  # Batch Normalization
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3.float()) + self.G_b3.float())  # [0,1] normalized Output

        self.batch_mean1 = self.G_bn1.running_mean
        self.batch_var1 = self.G_bn1.running_var


        self.batch_mean2 = self.G_bn2.running_mean
        self.batch_var2 = self.G_bn2.running_var


        return G_prob



def loss(X, M, Noise):

    G_sample = generator(Noise, M)
    
    return  torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M), G_sample


generator = Generator(Dim, H_Dim1, H_Dim2)
optimizer = torch.optim.Adam(params=generator.parameters())


for it in tqdm(range(epoch)):
    generator.train()

    #print('Initial weight & bias:', generator.weight.data, generator.bias.data)
    batch_no = 0
    print('======Running first pass======')
    train_iterator = tqdm(train_loader, desc='Training Batch', leave=False)
    for X_mb, M_mb in train_iterator:

        generator.G_bn1.track_running_stats = False 
        generator.G_bn2.track_running_stats = False

        Z = sample_Z(X_mb.shape[0], Dim)
        Noise = M_mb * X_mb + (1-M_mb) * Z

        optimizer.zero_grad()

        G_loss = loss(X=X_mb, M=M_mb, Noise=Noise)[0]
        G_loss.backward()
        optimizer.step()
        batch_no += 1
        
        train_iterator.set_postfix({'Train Loss': G_loss.item()})
        print("Batch:",batch_no)
        print('1st BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(generator.batch_mean1), torch.mean(generator.batch_var1)))
        print('2nt BatchNormMean: {:.4} Var:{:.4}'.format(torch.mean(generator.batch_mean2), torch.mean(generator.batch_var2)), end='\n\n')

    print('======Running second pass======')
    batch_no = 0
    for X_mb, M_mb in train_iterator:

        generator.G_bn1.track_running_stats = True 
        generator.G_bn2.track_running_stats = True
        
        Z = sample_Z(X_mb.shape[0], Dim)
        Noise = M_mb * X_mb + (1-M_mb) * Z

        optimizer.zero_grad()

        Imp = loss(X=X_mb, M=M_mb, Noise=Noise)[1]

        batch_no += 1

        print("Batch:",batch_no)
        print('1st BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(generator.batch_mean1), torch.mean(generator.batch_var1)))
        print('2nt BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(generator.batch_mean2), torch.mean(generator.batch_var2)), end='\n\n')
    
    print('Iter: {}'.format(it), end='\t')
    print('Train_loss: {:.4}'.format(np.sqrt(G_loss.item())), end='\n\n\n')


# Evaluation

with torch.no_grad():
    generator.eval()
    MSE = []
    for X_mb, M_mb in test_loader:
        
        Z = sample_Z(X_mb.shape[0], Dim)
        Noise = M_mb * X_mb + (1-M_mb) * Z

        MSE_final, Sample = loss(X=X_mb, M=M_mb, Noise=Noise)
        MSE.append(MSE_final)

    #print([mse for mse in MSE_final]/len(MSE_final))

mse_final_mean = torch.mean(MSE_final)
rmse_final = torch.sqrt(mse_final_mean)

print('Final Test RMSE: {:.4f}'.format(rmse_final.item()))

# imputed_data = M_mb * X_mb + (1-M_mb) * Sample
# print("Imputed test data:")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

# if use_gpu is True:
#     print(imputed_data.cpu().detach().numpy())
# else:
#     print(imputed_data.detach().numpy())