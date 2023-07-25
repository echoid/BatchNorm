import torch
import os
import sys
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from GAIN_imputer_utility import xavier_init,MyDataset,preprocess,load_dataloader
from sklearn.impute import SimpleImputer
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
from MNAR.missing_process.block_rules import *


#dataset_file = "banknote"#'concrete_compression', "wine_quality_red","wine_quality_white"
  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset
missing_type = "quantile"

  # set it to True to use GPU and False to use CPU
use_BN = True
states = [False,True]

dataset_file = 'california'

missing_rule = ["Q1_complete","Q1_partial","Q2_complete","Q2_partial","Q3_complete","Q3_partial","Q4_complete","Q4_partial",
"Q1_Q2_complete","Q1_Q2_partial","Q1_Q3_complete","Q1_Q3_partial","Q1_Q4_complete","Q1_Q4_partial","Q2_Q3_complete","Q2_Q3_partial",
"Q2_Q4_complete","Q2_Q4_partial","Q3_Q4_complete","Q3_Q4_partial"]

missing_rule = ["C0_lower","C0_upper","C0_double","C1_lower","C1_upper","C1_double", 
                "C2_lower","C2_upper","C2_double", "C3_lower","C3_upper", "C3_double",
                "C4_lower","C4_upper","C4_double","C5_lower","C5_upper","C5_double",
                "C6_lower","C6_upper","C6_double","C7_lower","C7_upper","C7_double",
]

missing_type = "BN"




#%% System Parameters
batch_size = 64
epoch = 500



class Simple_imputer(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = nn.Linear(dim, dim)

    def forward(self, data, m):
        imputed_data = torch.sigmoid(self.linear(data))

        return imputed_data

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

def get_dataset_loaders(trainX, train_Mask,train_input,testX,test_Mask,test_input,train_H,test_H):

    train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input,train_H), MyDataset(testX, test_Mask, test_input,test_H)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader , test_loader

def loss(truth, mask, data,imputer):
    generated = imputer(data, mask)
    
    RMSE = torch.sqrt(torch.sum((truth - generated) ** 2 * (1 - mask))/torch.sum(1 - mask))

    return  RMSE, generated

def impute_with_prediction(original_data, mask, prediction):

    return mask * original_data + (1 - mask) * prediction









def check_and_fill_nan(array, reference_array):
    num_rows, num_cols = array.shape

    for col_idx in range(num_cols):
        if np.all(np.isnan(array[:, col_idx])):
            array[0, col_idx] = np.nanmean(reference_array[:, col_idx])

    return array



def run(dataset_file,missing_rule, use_BN):
    
   

    Imputer_RMSE = []
    baseline_RMSE = []
    
    for rule_name in missing_rule:
        print(dataset_file,rule_name,use_BN,states)
        trainX, testX, train_Mask, test_Mask, train_input, test_input, No, Dim,train_H, test_H = load_dataloader(dataset_file,missing_type, rule_name)

    
        # train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input,train_H), MyDataset(testX, test_Mask, test_input,test_H)

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        imputer = Imputation_model(Dim, Dim, Dim, use_BN)
        #imputer = Simple_imputer(Dim)
        optimizer = torch.optim.Adam(params=imputer.parameters())

        train_loader , test_loader = get_dataset_loaders(trainX, train_Mask,train_input, testX, test_Mask, test_input,train_H,test_H)


        for it in tqdm(range(epoch)):
            imputer.train()
            total_loss = 0
            batch_no = 0
            for truth_X, mask, data_X,x_hat in train_loader:
                batch_no += 1

                # print("======Batch {} Start======".format(batch_no))
                # print('Running first pass:')

                set_all_BN_layers_tracking_state(imputer,True)

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
            RMSE_total = []
            for truth_X, mask, data_X , x_hat in test_loader:

                RMSE, prediction =  loss(truth=truth_X, mask=mask, data=data_X,imputer = imputer)
                imputed_data = impute_with_prediction(truth_X, mask, prediction)
                RMSE_total.append(RMSE)


        RMSE_tensor = torch.tensor(RMSE_total)
        rmse_final = torch.mean(RMSE_tensor)

        Imputer_RMSE.append(round(rmse_final.item(),5))

        print('Final Test RMSE: {:.4f}'.format(rmse_final.item()))

        ###################Baseline###############################
 
        train_nan = trainX.copy()
        test_nan = testX.copy()
        train_nan[train_Mask == 0] = np.nan
        test_nan[test_Mask == 0] = np.nan

        train_nan = check_and_fill_nan(train_nan,trainX)

        # ------------------------------------------------------------------------------
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(train_nan)
        train_imp = imp.transform(train_nan)
        test_imp = imp.transform(test_nan)
        #train_rmse = np.sqrt(np.sum((trainX - train_imp) ** 2 * (1 - train_mask)) / np.sum(1 - train_mask))
        test_rmse = np.sqrt(np.sum((testX - test_imp) ** 2 * (1 - test_Mask)) / np.sum(1 - test_Mask))


        print("Mean Imputer test_rmse:",test_rmse)


        baseline_RMSE.append(round(test_rmse,5))


    result = pd.DataFrame({"Missing_Rule":[rule_name for rule_name in missing_rule],"Imputer RMSE":Imputer_RMSE,"Baseline RMSE":baseline_RMSE})
    result.to_csv("results/GAIN_imputer/{}_2pass.csv".format(dataset_file),index=False)


run(dataset_file,missing_rule,use_BN)