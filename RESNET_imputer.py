
import torch
import os
import sys
import numpy as np
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from RESNET_imputer_utility import MyDataset,load_dataloader,ResNetImputer,ResidualBlock,weights_init,set_all_BN_layers_tracking_state,get_dataset_loaders
from RESNET_imputer_utility import loss,check_and_fill_nan
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

dataset_file = sys.argv[1]

missing_rule = ["Q1_complete"]
#te","Q2_partial","Q3_complete","Q3_partial","Q4_complete","Q4_partial",
# "Q1_Q2_complete","Q1_Q2_partial","Q1_Q3_complete","Q1_Q3_partial","Q1_Q4_complete","Q1_Q4_partial","Q2_Q3_complete","Q2_Q3_partial",
# "Q2_Q4_complete","Q2_Q4_partial","Q3_Q4_complete","Q3_Q4_partial"]


#%% System Parameters
batch_size = 64
epoch = 200





def impute_with_prediction(original_data, mask, prediction):

    return mask * original_data + (1 - mask) * prediction











def run(dataset_file,missing_rule, use_BN):
    
   

    Imputer_RMSE = []
    baseline_RMSE = []
    
    for rule_name in missing_rule:
        print(dataset_file,rule_name,use_BN,states)
        trainX, testX, train_Mask, test_Mask, train_input, test_input, No, Dim = load_dataloader(dataset_file,missing_type, rule_name)

    
        train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input), MyDataset(testX, test_Mask, test_input)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        #imputer = Imputation_model(Dim, Dim, Dim, use_BN)
        print(Dim)
        imputer = ResNetImputer(ResidualBlock,[3, 4, 6, 3],Dim)
        imputer.apply(weights_init) 

        optimizer = torch.optim.Adam(params=imputer.parameters())

        #train_loader , test_loader = get_dataset_loaders(trainX, train_Mask,train_input, testX, test_Mask, test_input)


        for it in tqdm(range(epoch)):
            imputer.train()
            total_loss = 0
            batch_no = 0
            for truth_X, mask, data_X in train_loader:
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
            for truth_X, mask, data_X in test_loader:

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
    result.to_csv("results/GAIN_imputer/{}_32_2pass.csv".format(dataset_file),index=False)


run(dataset_file,missing_rule,use_BN)




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img
