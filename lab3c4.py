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
        output = F.log_softmax(self.linear3(z))
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


def partition_dataset(dataset):
   
    size = dist.get_world_size()-1
    bsz = batch_size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank()-1)
    data_set = DataLoader(partition,
                            batch_size=bsz,
                            shuffle=True)
    return data_set, bsz



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




def runWorker(dataset, criterion, group, model):


    torch.manual_seed(1234)
    

    size = dist.get_world_size()
    rank = dist.get_rank() 

    epoch_loss = 0.0
    numberOfSamples = 0

    train_set, bsz = partition_dataset(dataset)


    num_batches = ceil(len(train_set.dataset) / float(bsz))
    print("started ",rank)
    dist.send(tensor = torch.Tensor([0]),dst = 0)
    for param in model.parameters():
        dist.recv(tensor = param.data, src = 0)
    dist.barrier(group)
    for epoch in range(epochs):
        epoch_loss = 0.0
        numberOfSamples = 0
        for batch_idx, (data, target) in enumerate(train_set):
            numberOfSamples += data.size()[0]
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            dist.send(tensor = torch.Tensor([rank]),dst = 0)
            for param in model.parameters():
                dist.send(tensor = param.grad.data, dst = 0)
            for param in model.parameters():
                dist.recv(tensor = param.data, src = 0)
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_set.dataset), 100. * batch_idx / len(train_set), loss.item()))
        dist.barrier(group)
        dist.send(tensor = torch.Tensor([0]),dst = 0)
        for param in model.parameters():
            dist.recv(tensor = param.data, src = 0)
        
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
    dist.send(tensor = torch.Tensor([-1]),dst = 0)

        

    print('Rank ', dist.get_rank(), ', epoch_loss ', epoch_loss, ', number of samples ', numberOfSamples)

    loss_w = torch.Tensor([epoch_loss * numberOfSamples])
    numberOfSamples = torch.Tensor([numberOfSamples])
    dist.all_reduce(loss_w, op=dist.reduce_op.SUM, group=group)
    dist.all_reduce(numberOfSamples, op=dist.reduce_op.SUM, group=group)

    if rank == 1:
        print("Final Weighted Loss - ",(loss_w/numberOfSamples))

def runServer(model, optimizer, criterion):
    
    numberOfTimes = dist.get_world_size()-1
    for param in model.parameters():
        param.sum().backward()
    tag = torch.zeros(1)
    while True:
        src = dist.recv(tensor = tag)
        # print("Reached ", src)
        if tag[0] == 0:
            for param in model.parameters():
                dist.send(tensor = param.data, dst = src)
        elif tag[0] == -1:
            numberOfTimes -= 1
            if numberOfTimes == 0:
                # print("------------- Breaking ----------------")
                break
        else:
            for param in model.parameters():
                dist.recv(tensor = param.grad.data, src = src)
            optimizer.step()
            for param in model.parameters():
                dist.send(tensor = param.data, dst = src)







def main():


    data_transform = transforms.Compose([
                                transforms.Resize((32,32)),
                                transforms.ToTensor()
                            ])
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum = 0.9)
    
    train_dataset = data(csv_file = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train.csv', root_dir = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train-jpg/',transform = data_transform)
    dist.init_process_group(backend="mpi")
    group = dist.new_group([i for i in range(1, dist.get_world_size())])
    # size = dist.get_world_size()
    # rank = dist.get_rank() 
    if dist.get_rank() != 0:
        runWorker(train_dataset, criterion, group, net)
    else:
        runServer(net, optimizer, criterion)
    # train_set, bsz = partition_dataset(train_dataset)
    # t0 = time.monotonic()
    # run(dist.get_rank() , dist.get_world_size(), train_set,bsz, net, optimizer, criterion)
    # t0 = time.monotonic()-t0
    # if dist.get_rank() == 0:
    #     print("Final Weighted Loss - ",(weighted_loss/numberOfSamples))
    #     print("The time is - ",t0)
    # test_dataset = data(csv_file = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/test.csv', root_dir = '/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train-jpg/',transform = data_transform)


if __name__ == "__main__":
    main()
