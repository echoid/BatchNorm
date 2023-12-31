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


    # train_H = sample_M(Xtrain.shape[0], D, 1-p_hint)
    # train_H = Xtrain_mask * train_H


    # test_H = sample_M(Xtest.shape[0], D, 1-p_hint)
    # test_H = Xtest_mask * test_H


    return Xtrain, Xtest, Xtrain_mask, Xtest_mask , train_input, test_input , N, D


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)
    


# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n]) 



def check_and_fill_nan(array, reference_array):
    num_rows, num_cols = array.shape

    for col_idx in range(num_cols):
        if np.all(np.isnan(array[:, col_idx])):
            array[0, col_idx] = np.nanmean(reference_array[:, col_idx])

    return array


class MyDataset(Dataset):
    def __init__(self, X, M, input):
        self.X = torch.tensor(X).float()
        self.M = torch.tensor(M).float()
        self.input = torch.tensor(input).float()
        #self.h = torch.tensor(h).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.input[idx]#,self.h[idx]
    


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

def get_dataset_loaders(trainX, testX, train_Mask, test_Mask, train_input, test_input,batch_size = 128):

    train_dataset, test_dataset = MyDataset(trainX, train_Mask,train_input), MyDataset(testX, test_Mask, test_input)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader , test_loader

def loss(truth, mask, data,imputer):

    generated = imputer(data, mask)

    return  torch.mean(((1 - mask) * truth - (1 - mask) * generated) ** 2) / torch.mean(1 - mask), generated

def impute_with_prediction(original_data, mask, prediction):

    return mask * original_data + (1 - mask) * prediction

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNetImputer(nn.Module):
    def __init__(self, block, layers, input_dim):
        super(ResNetImputer, self).__init__()

        self.inplanes = input_dim

        self.fc1 = nn.Linear(input_dim * 2, 64)
        self.relu = nn.ReLU()

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1, input_dim=input_dim)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(512, input_dim)

    def _make_layer(self, block, planes, blocks, stride=1, input_dim=None):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Linear(self.inplanes, planes),
                nn.BatchNorm1d(planes)
            )

        layers = []
        layers.append(block(input_dim, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.unsqueeze(2)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)

        return x
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