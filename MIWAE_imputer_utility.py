#!pip3 install --user --upgrade scikit-learn # We need to update it to run missForest

import torch
import torch.nn as nn
import numpy as np
import scipy.stats
import scipy.io
import scipy.sparse
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributions as td
import torch
from torch import nn, optim
from torch.nn import functional as F
import pickle
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, X_mask, X_hat_0, X_hat):
        self.X = X
        self.X_mask = X_mask
        self.X_hat_0 = X_hat_0
        self.X_hat = X_hat

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mask = self.X_mask[idx]
        x_hat_0 = self.X_hat_0[idx]
        x_hat = self.X_hat[idx]
        return x, mask, x_hat_0, x_hat
    


def rmse(xhat,xtrue,mask): # MSE function for imputations
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.sqrt(np.mean(((1 - mask) * xtrue - (1 - mask) * xhat) ** 2) / np.mean(1 - mask))


def set_BN_layers_tracking_state(model, tracking_state):
    if hasattr(model, 'encoder'):
        for module in model.encoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = tracking_state

    if hasattr(model, 'decoder'):
        for module in model.decoder.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = tracking_state

def weights_init(layer):
  if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)





class MiwaeImputer(nn.Module):
    def __init__(self, p, d, h, K, cuda=False, use_bn = [True,True]):
        super(MiwaeImputer, self).__init__()
        self.cuda = cuda
        self.d = d
        self.p = p
        self.h = h
        self.K = K
        self.use_bn = use_bn

        if self.cuda:
            self.p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(), scale=torch.ones(d).cuda()), 1)
        else:
            self.p_z = td.Independent(td.Normal(loc=torch.zeros(d), scale=torch.ones(d)), 1)
        if use_bn[0]:
            self.encoder = nn.Sequential(
            nn.Linear(p, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, 2 * d),
            )

        else:
            self.encoder = nn.Sequential(
                nn.Linear(p, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 2 * d),
            )

        self.encoder = self.encoder.apply(weights_init)

        if use_bn[1]:
            self.decoder = nn.Sequential(
            nn.Linear(d, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, 3 * p),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, 3 * p),
            )

        self.decoder = self.decoder.apply(weights_init)

        if self.cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def forward(self, x, mask, L):
        batch_size = x.shape[0]
        out_encoder = self.encoder(x)
        q_zgivenxobs = td.Independent(
            td.Normal(loc=out_encoder[..., :self.d], scale=torch.nn.Softplus()(out_encoder[..., self.d:(2 * self.d)])),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.d])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.p]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.p:(2 * self.p)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2 * self.p):(3 * self.p)]) + 3

        if self.cuda:
            data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1]).cuda()
            tiledmask = torch.Tensor.repeat(mask, [L, 1]).cuda()
        else:
            data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1])
            tiledmask = torch.Tensor.repeat(mask, [L, 1])

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape([-1, 1]),
            scale=all_scales_obs_model.reshape([-1, 1]),
            df=all_degfreedom_obs_model.reshape([-1, 1])
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, self.p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([L, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(
            td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),
            1
        )

        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq, 0)
        xms = xgivenz.sample().reshape([L, batch_size, self.p])
        xm = torch.einsum('ki,kij->ij', imp_weights, xms)

        return xm
    

    def miwae_loss(self, iota_x, mask):
        batch_size = iota_x.shape[0]
        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(
            td.Normal(loc=out_encoder[..., :self.d], scale=torch.nn.Softplus()(out_encoder[..., self.d:(2 * self.d)])),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([self.K])
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.d])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.p]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.p:(2 * self.p)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2 * self.p):(3 * self.p)]) + 3

        if self.cuda:
            data_flat = torch.Tensor.repeat(iota_x, [self.K, 1]).reshape([-1, 1]).cuda()
            tiledmask = torch.Tensor.repeat(mask, [self.K, 1]).cuda()
        else:
            data_flat = torch.Tensor.repeat(iota_x, [self.K, 1]).reshape([-1, 1])
            tiledmask = torch.Tensor.repeat(mask, [self.K, 1])

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape([-1, 1]),
            scale=all_scales_obs_model.reshape([-1, 1]),
            df=all_degfreedom_obs_model.reshape([-1, 1])
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K * batch_size, self.p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([self.K, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(
            td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),
            1
        )

        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq, 0)
        xms = xgivenz.sample().reshape([self.K, batch_size, self.p])
        xm = torch.einsum('ki,kij->ij', imp_weights, xms)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

        return neg_bound


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
    Xval = observed_values[valid_index]

    Xtrain_mask = gt_masks[train_index]
    Xtest_mask = gt_masks[test_index]
    Xval_mask = gt_masks[valid_index]

    X_train_miss = np.copy(Xtrain)
    X_train_miss[Xtrain_mask == 0] = np.nan # np.nan fill in

    X_test_miss = np.copy(Xtest)
    X_test_miss[Xtest_mask == 0] = np.nan # np.nan fill in


    x_train_hat_0 = np.copy(X_train_miss)
    x_test_hat_0 = np.copy(X_test_miss)

    x_train_hat_0[np.isnan(X_train_miss)] = 0
    x_train_hat = np.copy(x_train_hat_0)

    x_test_hat_0[np.isnan(X_test_miss)] = 0
    x_test_hat = np.copy(x_test_hat_0)

    return Xtrain, Xtest, Xtrain_mask, Xtest_mask, N, D, x_train_hat_0, x_test_hat_0, x_train_hat, x_test_hat,X_train_miss,X_test_miss