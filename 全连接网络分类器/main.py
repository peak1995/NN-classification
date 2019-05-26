import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pyplot

from Model import nnNet

DATAPATH = '/Users/peak/Desktop/作业5/dataset/'
BATCHSIZE = 256
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
def load_data(data_path):
    with open(data_path+'trainX.txt', 'r') as f:
        train_x = np.array(list(map(float, f.read().split())))
    with open(data_path+'trainY.txt', 'r') as f:
        train_y = np.array(list(map(int, f.read().split())))
    with open(data_path+'testX.txt', 'r') as f:
        test_x = np.array(list(map(float, f.read().split())))
    with open(data_path+'testY.txt', 'r') as f:
        test_y = np.array(list(map(int, f.read().split())))
    return torch.from_numpy(train_x.reshape(-1, 13)), torch.from_numpy(train_y-1), torch.from_numpy(test_x.reshape(-1, 13)), torch.from_numpy(test_y-1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset) / BATCHSIZE
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) 

def main():
    train_x, train_y, test_x, test_y = load_data(DATAPATH)
    train_x = train_x.float()
    train_y = train_y.long()
    test_x = test_x.float()
    test_y = test_y.long()
    print(train_x.size(), train_y.size(), test_x.size(), test_y.size())
    print(train_x[0])
    print(train_y[0])
    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    train_loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = BATCHSIZE,
        shuffle = True)
    test_loader = Data.DataLoader(
        dataset = test_dataset,
        batch_size = BATCHSIZE,
        shuffle = True)

    model = nnNet(1, [13, 100, 19], 'relu', DEVICE).to(DEVICE)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)

if __name__ == '__main__':
    main()