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
from tensorboardX import SummaryWriter

writer = SummaryWriter()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_np = train.to_numpy()

# checking ratio between classes
# print(np.array([np.sum(train_np[:,0] ==0),np.sum(train_np[:,0] ==1),np.sum(train_np[:,0] ==2),np.sum(train_np[:,0] ==3),np.sum(train_np[:,0] ==4)])/train_np.shape[0]*100)


train, test = train_test_split(train_np,test_size = 0.2, random_state=456)
trainfeatures = train[:, 1:]
testfeatures = test[:,1:]
trainlabels = train[:, 0]
testlabels = test[:,0]

# checking covarriance
# plt.matshow(np.corrcoef(trainfeatures,rowvar = False))

scaler = StandardScaler()
scaler.fit(trainfeatures)


trainfeatures = scaler.transform(trainfeatures)

pca = PCA(n_components = 120)
pca.fit(trainfeatures)
sum(pca.explained_variance_ratio_)
trainfeatures = pca.transform(trainfeatures)
plt.matshow(np.corrcoef(trainfeatures,rowvar = False))

testfeatures = scaler.transform(testfeatures)
testfeatures = pca.transform(testfeatures)
data = np.concatenate((trainfeatures, trainlabels.reshape((-1, 1))), axis = 1)
data = torch.from_numpy(data)
testdata = np.concatenate((testfeatures,testlabels.reshape((-1,1))),axis = 1)
testdata = torch.from_numpy(testdata)
train_loader = torch.utils.data.DataLoader(data, batch_size = 600)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv1d(1, 32, 3, 1,padding=1)
        # self.conv2 = nn.Conv1d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(120, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 5)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, 2)
        # x = x.view(-1, 64*5)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
trainloss= []
testloss = []
testaccuracy = []
epochs = 1500
optimizer = optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-5)
lossfunct = nn.CrossEntropyLoss()

for epoch in range(1, epochs + 1):
    # training mode (grads flow)
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device).float()
        # zero-ing existing gradients
        optimizer.zero_grad()
        # forward pass
        output = model((data[:,:-1]))
        output.type()
        # loss
        loss = lossfunct(output, data[:,-1].long())
        # backward pass
        loss.backward()
        # weight update
        optimizer.step()
        if epoch % 10 == 0:
            # storing training loss
            trainloss.append(loss)
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                output = model(((testdata[:,:-1]).float()))
                # getting error
                test_loss += lossfunct(output, testdata[:,-1].long()).item()
                testloss.append(test_loss)
                # getting pred
                pred = output.argmax(dim=1, keepdim=True)
                # accuracy
                correct += pred.eq(testdata[:,-1].long().view_as(pred)).sum().item()
                testaccuracy.append(correct / testdata.shape[0])
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
                test_loss, correct, testdata.shape[0],
                100. * correct / testdata.shape[0]))
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


plt.plot(testloss, label='Test')
plt.plot(trainloss, label='Train')
plt.plot(testaccuracy, label="Accuracy")
plt.legend()