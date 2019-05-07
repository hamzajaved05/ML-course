from sklearn.neighbors import DistanceMetric
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader
import torchvision
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hypertools as hyp
# from tensorboardX import SummaryWriter

# writer = SummaryWriter()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_np = train.to_numpy()
x = [train_np[i,1:] for i, j  in enumerate(train_np[:,0]) if j == 1]

# dist = DistanceMetric.get_metric('mahalanobis')
# y = dist.pairwise(x)

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(train))
threshold = 3
print(np.where(z > 3))