import time
import os
import sys
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from math import ceil
from random import Random
import torch.distributed as dist
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Net(nn.Module):

    def __init__(self, i1, o1, o2, o3):
        super(Net, self).__init__()
        self.i1 = i1
        self.o1 = o1
        self.o2 = o2
        self.o3 = o3
        self.fc1 = nn.Linear(i1,o1)
        self.fc2 = nn.Linear(o1,o2)
        self.fc3 = nn.Linear(o2,o3)

    def forward(self, x):
        x = x.view(-1, self.i1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),0)
        return x


class Partition(Dataset):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(Dataset):

    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class kaggleDataset(Dataset):

    def __init__(self, csvPath, imagesPath, transform=None):
    
        self.data = pd.read_csv(csvPath)
        self.imagesPath = imagesPath
        self.transform = transform

        self.imagesData = self.data['image_name']
        self.labelsData = self.data['tag'].astype('int')

    def __getitem__(self, index):
        imageName = os.path.join(self.imagesPath,self.data.iloc[index, 0])
        image = Image.open(imageName + '.jpg')
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labelsData[index]
        return image, label

    def __len__(self):
        return len(self.data)


def partition_dataset(dataLabels, imagesPath, transformations, batchSize):

    dataset = kaggleDataset(dataLabels,imagesPath,transformations)
    size = dist.get_world_size()
    bsz = batchSize
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=1)
    return train_set, bsz


def average_gradients(model):

    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def run(rank, size, model, optimizer, criterion, epochs, loader, bsz):
    
    torch.manual_seed(1234)
    epoch_loss = 0.0
    numberOfSamples = 0
    num_batches = ceil(len(loader.dataset) / float(bsz))
    for epoch in range(epochs):
        epoch_loss = 0.0
        numberOfSamples = 0
        for batch_idx, (data, target) in enumerate(loader):
            numberOfSamples += data.size()[0]
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))

        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

    weighted_loss = torch.Tensor(epoch_loss * numberOfSamples)
    numberOfSamples = torch.Tensor(numberOfSamples)
    dist.all_reduce(weighted_loss, op=dist.reduce_op.SUM, group=0)
    dist.all_reduce(numberOfSamples, op=dist.reduce_op.SUM, group=0)

    return weighted_loss, numberOfSamples



def main(rank, wsize):

    batchSize = 100
    epochs = 1
    learningRate = 0.01
    momentum = 0.9
    i1 = 3072
    o1 = 1024
    o2 = 256
    o3 = 17
    numWorkers = 1

    net = Net(i1,o1,o2,o3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = learningRate, momentum = momentum)

    imagesPath = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train-jpg/'
    trainData = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train.csv'
    testData = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/test.csv'
    print('Declared Net and set paths')

    transformations = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

    trainLoader, bszTrain = partition_dataset(trainData, imagesPath, transformations, batchSize)
    testLoader, bszTest = partition_dataset(testData, imagesPath, transformations, batchSize)
    
    print('Created datasets')

    weighted_loss, numberOfSamples = run(rank, wsize, net, optimizer, criterion, epochs, trainLoader, bszTrain)

    if rank == 0:
        print("Final Weighted Loss - ",(weighted_loss/numberOfSamples))



if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("ERROR")
        sys.exit(1)
    
    dist.init_process_group(backend="mpi", world_size=4)
    rank = dist.get_rank()
    wsize = dist.get_world_size()

    main(rank, wsize)