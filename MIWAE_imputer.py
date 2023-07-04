#!pip3 install --user --upgrade scikit-learn # We need to update it to run missForest

import torch
import torch.nn as nn
import numpy as np
import os
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch.distributions as td
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
    
from MIWAE_imputer_utility import *


parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
from MNAR.missing_process.block_rules import *

dataset_file = sys.argv[1]

missing_rule = ["Q1_complete","Q1_partial","Q2_complete","Q2_partial","Q3_complete","Q3_partial","Q4_complete","Q4_partial",
"Q1_Q2_complete","Q1_Q2_partial","Q1_Q3_complete","Q1_Q3_partial","Q1_Q4_complete","Q1_Q4_partial","Q2_Q3_complete","Q2_Q3_partial",
"Q2_Q4_complete","Q2_Q4_partial","Q3_Q4_complete","Q3_Q4_partial"]



h = 128 # number of hidden units in (same for all MLPs)
d = 1 # dimension of the latent space
K = 20 # number of IS during training
bs = 64 # batch size
n_epochs = 2002


#cuda = torch.cuda.is_available()
cuda = False
use_bn = [True,True]




def run(dataset_file,missing_name):


    Xtrain, Xtest, Xtrain_mask, Xtest_mask, n, \
        p, x_train_hat_0, x_test_hat_0, x_train_hat,\
            x_test_hat,X_train_miss,X_test_miss = load_dataloader(dataset_file,missing_type = "quantile", \
                                        missing_name = missing_name,seed = 1)




    # Create training dataset and dataloader
    train_dataset = MyDataset(Xtrain, Xtrain_mask, x_train_hat_0, x_train_hat)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    # Create testing dataset and dataloader
    test_dataset = MyDataset(Xtest, Xtest_mask, x_test_hat_0, x_test_hat)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


    mse_train=np.array([])



    # Create MiwaeImputer instance
    MIWAE = MiwaeImputer(p, d, h, K, cuda, use_bn)

    # Define optimizer
    optimizer = optim.Adam(MIWAE.parameters(), lr=1e-3)

    # Training loop
    for epoch in tqdm(range(n_epochs)):
        MIWAE.train()

        batch_no = 0
        for x, mask, x_hat_0, x_hat in train_loader:
            batch_no += 1
            optimizer.zero_grad()
            MIWAE.zero_grad()
            if cuda:
                b_data = x_hat_0.float().cuda()
                b_mask = mask.float().cuda()
                x_hat = x_hat.float().cuda()
            else:
                b_data = x_hat_0.float()
                b_mask = mask.float()
                x_hat = x_hat.float()
            

            set_BN_layers_tracking_state(MIWAE, [False, False])

            #x_hat[~mask] = torch.from_numpy(MIWAE(b_data, b_mask, L=10).detach().numpy()[~mask])

            x_imp = torch.from_numpy(MIWAE(b_data, b_mask, L=10).detach().numpy())
            mask_bool = mask.bool()
            x_hat[~mask_bool] = x_imp[~mask_bool]

            loss = MIWAE.miwae_loss(iota_x=b_data, mask=b_mask)
            loss.backward()
            optimizer.step()


            set_BN_layers_tracking_state(MIWAE, [True, True])

            _ = MIWAE(x_hat.float(), b_mask.float(), L=10).cpu().detach().numpy()


        if epoch % 100 == 1:
            print('Epoch %g' % epoch)
            if cuda:
                print(
                    'MIWAE likelihood bound  %g' % (
                            -np.log(K) - MIWAE.miwae_loss(iota_x=b_data.float().cuda(),
                                                        mask=b_mask.float().cuda()).cpu().data.numpy()
                    )
                )
            else:
                print(
                    'MIWAE likelihood bound  %g' % (
                            -np.log(K) - MIWAE.miwae_loss(iota_x=b_data.float(),
                                                        mask=b_mask.float()).data.numpy()
                    )
                )

            x_train_hat[~Xtrain_mask] = MIWAE.forward(torch.from_numpy(x_train_hat_0).float(), torch.from_numpy(Xtrain_mask).float(), L=10).data.numpy()[~Xtrain_mask]
            err = np.array([rmse(x_train_hat, Xtrain, Xtrain_mask)])
            mse_train = np.append(mse_train, err, axis=0)
            print('Imputation RMSE  %g' % err)
            print('-----')

    #######################################################################

    missforest = IterativeImputer(max_iter=20, estimator=ExtraTreesRegressor(n_estimators=100))
    iterative_ridge = IterativeImputer(max_iter=20, estimator=BayesianRidge())
    mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Train
    missforest.fit(X_train_miss)
    iterative_ridge.fit(X_train_miss)
    mean_imp.fit(X_train_miss)


    # Train set Evaluation
    xhat_mf_train = missforest.transform(X_train_miss)
    xhat_ridge_train = iterative_ridge.transform(X_train_miss)
    xhat_mean_train = mean_imp.transform(X_train_miss)


    # Test set Evaluation
    xhat_mf = missforest.transform(X_test_miss)
    mf_test = rmse(xhat_mf, Xtest, Xtest_mask)

    xhat_ridge = iterative_ridge.transform(X_test_miss)
    ridge_test = rmse(xhat_ridge, Xtest, Xtest_mask)

    xhat_mean = mean_imp.transform(X_test_miss)
    mean_test = rmse(xhat_mean, Xtest, Xtest_mask)


    MIWAE.eval()
    x_test_hat[~Xtest_mask] = MIWAE.forward(torch.from_numpy(x_test_hat_0).float(), torch.from_numpy(Xtest_mask).float(), L=10).data.numpy()[~Xtest_mask]
    MIWAE_test = rmse(x_test_hat, Xtest, Xtest_mask)

    print("MF Test RMSE:",mf_test)
    print("Ridge Test RMSE:",ridge_test)
    print("Mean Test RMSE:",mean_test)
    print("MIWAE Test RMSE:",MIWAE_test)



    # plt.plot(range(1,n_epochs,100),mse_train,color="blue")
    # plt.axhline(y=rmse(xhat_mf_train, Xtrain, Xtrain_mask),  linestyle='-',color="red")
    # plt.axhline(y=rmse(xhat_ridge_train, Xtrain, Xtrain_mask),  linestyle='-',color="orange")
    # plt.axhline(y=rmse(xhat_mean_train, Xtrain, Xtrain_mask),  linestyle='-',color="green")
    # plt.legend(["MIWAE","missForest","Iterative ridge", "Mean imputation"])
    # plt.title("Train Set Imputation RMSE")
    # plt.xlabel("Epochs")
    # plt.savefig("image/{}_{}.png".format(dataset_file,missing_name))

    return [mf_test,ridge_test,mean_test,MIWAE_test
            ]



mf_result = []
ridge_result = []
mean_result = []
MIWAE_result = []

for missing_name in missing_rule:

    
    results = run(dataset_file, missing_name)

    mf_result.append(results[0])
    ridge_result.append(results[1])
    mean_result.append(results[2])
    MIWAE_result.append(results[3])


result = pd.DataFrame({"Missing_Rule":[rule_name for rule_name in missing_rule],
                       "MF RMSE":mf_result,
                       "Ridge RMSE":ridge_result,
                       "MEAN RMSE":mean_result
                       ,"MIWAE RMSE":MIWAE_result
                       })

result.to_csv("results/Second_{}_2Pass.csv".format(dataset_file),index=False)
    


