import torch
import numpy as np
import torch.nn as nn
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utility import xavier_init,MyDataset,preprocess


dataset_file = 'Spam.csv'  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset
use_gpu = False  # set it to True to use GPU and False to use CPU

if use_gpu:
    torch.cuda.set_device(0)
#%% System Parameters
batch_size = 32
epoch = 100


trainX, testX, train_Mask, test_Mask, train_input, test_input, No, Dim = preprocess(dataset_file)


class Imputation_model(nn.Module):
    def __init__(self, dim, hidden_dim1, hidden_dim2):
        super(Imputation_model, self).__init__()

        # self.No = No
        # self.Dim = Dim

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

    def forward(self, data, mask):
        inputs = torch.cat(dim=1, tensors=[data, mask])  # Mask + Data Concatenate
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

def set_all_BN_layers_tracking_state(model, state):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.track_running_stats = state

def get_dataset_loaders(trainX, testX, train_Mask, test_Mask, train_input, test_input):

    train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input), MyDataset(testX, test_Mask, test_input)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader , test_loader

def loss(truth, mask, data, imputer):

    generated = imputer(data, mask)

    return  torch.mean(((1 - mask) * truth - (1 - mask) * generated) ** 2) / torch.mean(1 - mask), generated

def impute_with_prediction(original_data, mask, prediction):

    return mask * original_data + (1 - mask) * prediction




imputer = Imputation_model(Dim, Dim, Dim)
optimizer = torch.optim.Adam(params=imputer.parameters())

train_loader , test_loader = get_dataset_loaders(trainX, testX, train_Mask, test_Mask, train_input, test_input)


for it in tqdm(range(epoch)):
    imputer.train()
    total_loss = 0
    batch_no = 0
    for truth_X, mask, data_X in train_loader:
        batch_no += 1

        # print("======Batch {} Start======".format(batch_no))
        # print('Running first pass:')

        set_all_BN_layers_tracking_state(imputer,False)

        optimizer.zero_grad()

        Imputer_loss = loss(truth=truth_X, mask=mask, data=data_X, imputer = imputer )[0]
        total_loss +=Imputer_loss
        Imputer_loss.backward()
        optimizer.step()

        # print('1st BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(imputer.batch_mean1), torch.mean(imputer.batch_var1)))
        # print('2nt BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(imputer.batch_mean2), torch.mean(imputer.batch_var2)), end='\n\n')

        # print('Running second pass:')

        set_all_BN_layers_tracking_state(imputer,True)

        prediction = loss(truth=truth_X, mask=mask, data=data_X,imputer = imputer)[1]

        imputed_data = impute_with_prediction(truth_X, mask, prediction)

        _ = imputer(imputed_data, mask)


        # print('1st BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(imputer.batch_mean1), torch.mean(imputer.batch_var1)))
        # print('2nt BatchNorm Mean: {:.4} Var:{:.4}'.format(torch.mean(imputer.batch_mean2), torch.mean(imputer.batch_var2)), end='\n\n')


        # print("======Batch {} End======\n\n".format(batch_no))


    print('Iter: {}'.format(it), end='\t')
    print('Train_loss: {:.4}'.format(np.sqrt(total_loss.item()/batch_no)))


# Evaluation

with torch.no_grad():
    imputer.eval()
    MSE_total = []
    for truth_X, mask, data_X in test_loader:

        MSE, prediction =  loss(truth=truth_X, mask=mask, data=data_X,imputer = imputer )
        imputed_data = impute_with_prediction(truth_X, mask, prediction)
        MSE_total.append(MSE)
        #print('Test_loss: {:.4}'.format(np.sqrt(MSE.item())))

    #print([mse for mse in MSE_final]/len(MSE_final))

MSE_tensor = torch.tensor(MSE_total)
rmse_final = torch.sqrt(torch.mean(MSE_tensor))

print('Final Test RMSE: {:.4f}'.format(rmse_final.item()))