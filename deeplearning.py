'''
Yiming Ge

This module contains deeplearning network functions in this program.
'''
import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from classification import*
import torch.utils.data as data_utils

n_epochs = 10
learning_rate = 0.01
momentum = 0.5
log_interval = 2


class StockDataset(Dataset):
    def __init__(self, df, transform=None, target_transfrom=None):
        features = df.copy()
        features = features.drop("Target", axis = "columns")
        # normalize
        data = (features - features.mean()) / (features.max() - features.min())
        # append the return signal back
        data = data.join(df['Target'])
        # print(df.head)
        # transfer it to numpy array
        self.final_data = data.to_numpy()
       
        # make the whole data set the same type
        self.final_data = self.final_data.astype(float)

        print("Finished data modifying")
        print(self.final_data.shape)

        self.transform = transform
        self.target_transform = target_transfrom

    def __len__(self):
        # return how many data points we have
        return self.final_data.shape[0]

    def __getitem__(self, idx):
        # return one example of the data which is specified by index
        tdata = self.final_data[idx, :-1]
        tdata = torch.from_numpy(tdata).type(torch.float32)
        label = int(self.final_data[idx, -1])

        if self.transform:
            pass
        if self.target_transform:
            pass
        return tdata, label



def get_tensor(data):
  train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=100, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    data,
    batch_size=100, shuffle=True)
  
  return train_loader, test_loader

# bulid full connected layers ANN
class FCNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        self.dropout_rate = dropout_rate
        super(FCNet, self).__init__()  # input is 18 features
        # create first linear layer
        self.fc1 = nn.Linear(5, 20)
        self.fc2 = nn.Linear(20, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
      

def train(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), '/Users/yiming/Downloads/results/ann_model.pth')
      torch.save(optimizer.state_dict(), '/Users/yiming/Downloads/results/ann_optimizer.pth')


def test(network, test_loader, test_losses, test_counter):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def test_given_range(model,test_loader,df,ticker1):
  # Make predictions on the data
  # model.load_state_dict(torch.load('/Users/yiming/Downloads/results/ann_model.pth'))
  model.eval()
  correct = 0
  for data, target in test_loader:
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
  temp = df.copy()
  temp['Return'] = df[ticker1] - df[ticker1].shift(1)
  test_actual_target = temp["Target"][-TEST_SIZE:-1]
  test_actual_return = temp["Return"][-TEST_SIZE:]
  # print(len(pred))
  # print(len(correct))
  # print(pred)
  draw_true_predict_classfication(test_actual_return, test_actual_target, pred.flatten().tolist(), "Fully Connected Network")
