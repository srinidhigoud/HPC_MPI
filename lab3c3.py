from __future__ import print_function
import sys
import os
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
from random import Random
import torch.distributed as dist
from PIL import Image
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time

#parameters
batch_size = 100   # input batch size for training
epochs = 5       # number of epochs to train
lr = 0.01
num_inputs_1 = 3072
num_outputs_1 = 1024
num_outputs_2 = 256
num_outputs_3 = 17

""" Dataset partitioning helper """

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(num_inputs_1, num_outputs_1)
        self.linear2 = nn.Linear(num_outputs_1, num_outputs_2)
        self.linear3 = nn.Linear(num_outputs_2, num_outputs_3)

    def forward(self, input):
        input = input.view(-1, num_inputs_1) # reshape input to batch x num_inputs
        z = F.relu(self.linear1(input))
        z = F.relu(self.linear2(z))
        output = F.log_softmax(self.linear3(z),0)
        return output


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
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


class data(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.file.iloc[idx, 0])
        image = Image.open(img_name+'.jpg')
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        labels = self.file.iloc[idx, 1].astype('int')
        return (image,labels)



def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data,
                        op=dist.reduce_op.SUM)
        param.grad.data /= size



def partition_dataset(dataset):
   
    size = dist.get_world_size()
    bsz = batch_size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    data_set = DataLoader(partition,
                            batch_size=bsz,
                            shuffle=True)
    return data_set, bsz




def run(rank, size, dataset_loader, batchSize, model, optimizer, criterion):

    torch.manual_seed(1234)
    

    size = dist.get_world_size()
    rank = dist.get_rank() 

    epoch_loss = 0.0
    numberOfSamples = 0

    # train_set, bsz = partition_dataset(dataset)
    # model = Net()
    # optimizer = optim.SGD(model.parameters(),
                            # lr=0.01, momentum=0.9)

    num_batches = ceil(len(dataset_loader.dataset) / float(batchSize))
    t0 = time.monotonic()
    for epoch in range(epochs):
        epoch_loss = 0.0
        numberOfSamples = 0
        for data, target in dataset_loader:
            numberOfSamples += data.size()[0]
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
    t0 = time.monotonic()-t0
    t0 /= epochs

        # print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)

    # print('Rank ', dist.get_rank(), ', epoch_loss ', epoch_loss / num_batches, ', number of samples ', numberOfSamples)

    # if rank == 1:
    #     print(t0)
    # print('Rank ', dist.get_rank(), ', epoch_loss ', epoch_loss/ num_batches, ', number of samples ', numberOfSamples)
    execTime = torch.Tensor([t0])
    loss_w = torch.Tensor([epoch_loss * numberOfSamples / num_batches])
    numberOfSamples = torch.Tensor([numberOfSamples])
    dist.all_reduce(loss_w, op=dist.reduce_op.SUM, group=0)
    dist.all_reduce(numberOfSamples, op=dist.reduce_op.SUM, group=0)
    dist.all_reduce(execTime, op=dist.reduce_op.SUM, group=0)
    if rank == 0:
        print("\n C3 \n")
        print(loss_w/numberOfSamples,',',execTime/size,' s')


def main():

    
    data_transform = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor()
                            ])
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum = 0.9)
    
    # net.cuda()
    train_dataset = data(csv_file = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train.csv', root_dir = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train-jpg/',transform = data_transform)
    dist.init_process_group(backend="mpi")

    train_set, bsz = partition_dataset(train_dataset)
    
    run(dist.get_rank() , dist.get_world_size(), train_set,bsz, net, optimizer, criterion)
    
    # if dist.get_rank() == 0:
    #     print("Final Weighted Loss - ",(weighted_loss/numberOfSamples), "The time is - ",t0)
        # print("The time is - ",t0)
    # test_dataset = data(csv_file = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/test.csv', root_dir = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train-jpg/',transform = data_transform)


if __name__ == "__main__":
    
    # dist.init_process_group(backend="mpi", world_size=int(sys.argv[1]))

    # dist.init_process_group(backend="mpi")
    # size = dist.get_world_size()
    # rank = dist.get_rank() 

    main()
