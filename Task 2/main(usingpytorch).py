"""
Author: Hamza
Dated: 07.04.2019
Project: IML Tasks
"""
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train = pd.read_csv('train.csv')
nptrain = train.to_numpy();
# train test split
train, test = train_test_split(nptrain[:,1:],test_size = 0.025, random_state=864)
# features separated for standaridization
trainfeatures = train[:, 1:]
testfeatures = test[:,1:]
trainlabels = train[:, 0]
testlabels = test[:,0]

# PCA using 14 PC
pca = PCA(n_components = 14)
# plt.matshow(np.corrcoef(trainfeatures,rowvar = False))
# plt.show()
# standardization and pca transform
scaler = StandardScaler()
scaler.fit(trainfeatures)
trainfeatures = scaler.transform(trainfeatures)
pca.fit(trainfeatures)
trainfeatures = pca.transform(trainfeatures)

# same transformation on test features
testfeatures = scaler.transform(testfeatures)
testfeatures = pca.transform(testfeatures)

# combine labels and features for a single iterable object
data = np.concatenate((trainfeatures, trainlabels.reshape((-1, 1))), axis = 1)
testdata = np.concatenate((testfeatures,testlabels.reshape((-1,1))),axis = 1)
testdata = torch.from_numpy(testdata)
train_loader = torch.utils.data.DataLoader(data , batch_size = 25)

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv1d(1, 32, 3, 1,padding=1)
        # self.conv2 = nn.Conv1d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(14, 14)
        self.fc2 = nn.Linear(14, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool1d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool1d(x, 2)
        # x = x.view(-1, 64*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
trainloss= []
testloss = []
testaccuracy = []
epochs = 2500
optimizer = optim.Adam(model.parameters(), lr=.0002, weight_decay=0.005)
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
        if epoch % 10 == 0 and batch_idx ==0:
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


scoredata = pd.read_csv('test.csv')
scoredata = scoredata.to_numpy()
scorefeatures = scoredata[:, 1:]
scorefeatures = scaler.transform(scorefeatures)
scorefeatures = pca.transform(scorefeatures)
output2 = model((torch.tensor(scorefeatures)).float())
pred2 = output2.argmax(dim=1, keepdim=True)
pred2 = pred2.numpy()
result = pd.DataFrame({"Id":np.linspace(2000, 4999, 3000), "y": pred2.reshape(-1)})
result.to_csv('result.csv', index = False, header = True)
plt.plot(testloss, label='Test')
plt.plot(trainloss, label='Train')
plt.plot(testaccuracy, label="Accuracy")
plt.legend()