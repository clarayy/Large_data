import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import random

import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ExampleNet import ExampleNet
from ExampleNet import CNN
from ExampleNet import CNN_500_part
# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
# tensor_x = torch.Tensor(my_x) # transform to torch tensor
# tensor_y = torch.Tensor(my_y)
N=214
fname = 'anu'+str(N)
bmname = fname+'_p0.5_m50'

def read_directory(directory_name):
    array_of_img = []  # this if for store all of the image data
    files = os.listdir(r"./"+directory_name)
    files.sort(key=lambda x: int(x[:-4]))
    #files.sort()
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename,-1)
        array_of_img.append(img)
        #print(img)
    array_of_img=np.array(array_of_img)
    return array_of_img

def pre_data(bmname):
    prefix = os.path.join('cnn_data_dw_1', bmname,bmname)
    filename_classlabel = prefix + '_graph_labels.txt'
    graph_label = []
    with open(filename_classlabel, 'r') as f1:
        for line in f1:
            graph_label.append(int(line.strip('\n')))
    tensor_label = torch.Tensor(graph_label).long()
    #adj_data = np.load(prefix+'_adj.npy')
    org_img_folder = prefix + '_adjdata'
    adj_data = read_directory(org_img_folder)
    tensor_data = torch.Tensor(adj_data).short()
    return tensor_data, tensor_label
#my_dataset = TensorDataset(tensor_data,tensor_label) # create your datset
#my_dataloader = DataLoader(my_dataset) # create your dataloader
#dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=4, shuffle=True)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10

    batch_size = 20

    #net = ExampleNet().to(device)
    net = CNN_500_part(N).to(device)
    criterion = nn.MSELoss(reduce = None, size_average = None)
    cross = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # train_data1, train_label1 = pre_data('BA300_SI_m50_train1')
    # train_data2, train_label2 = pre_data('BA300_SI_m50_train2')
    # train_data = np.vstack((train_data1, train_data2))
    # train_label1.extend(train_label2)
    # train_data = torch.Tensor(train_data).long()
    # train_label = torch.Tensor(train_label1).long()
    # train_data = torch.unsqueeze(train_data, dim=1).type(torch.FloatTensor)
    # train_dataset = TensorDataset(train_data, train_label)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_data, train_label = pre_data(bmname+'_train')
    train_data = torch.unsqueeze(train_data, dim=1).type(torch.FloatTensor)
    train_dataset = TensorDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data, test_label = pre_data(bmname+'_test')
    #test_data, test_label = pre_data('BA300_SI_m5_test')
    test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor)
    test_dataset = TensorDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for i in range(epochs):

        for j, (input, target) in enumerate(train_dataloader):
            input = input.to(device)
            output = net(input)
            target = torch.zeros(target.size(0), N).scatter_(1, target.view(-1, 1), 1).to(device)
            #loss = criterion(output, target)
            _, indices = torch.max(target, 1)
            loss = cross(output,indices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                losses.append(loss.float().detach().numpy())
                print("[epochs - {0} - {1}/{2}]loss: {3}".format(i, j, len(train_dataloader), loss.float()))
                plt.clf()
                #plt.plot(losses)
                plt.savefig('/home/iot/zcy/usb/copy/my_CNN/loss/'+bmname+'_loss.jpg')
                #plt.pause(0.01)
        with torch.no_grad():
            net.eval()
            correct = 0.
            total = 0.
            labels= []
            preds=[]
            for input, target in test_dataloader:
                input, target = input.to(device), target.to(device)
                output = net(input)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                accuracy = correct.float() / total
                labels.append(target.long().numpy())
                preds.append(predicted.long().numpy())
            net.train()
            print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))

            labels = np.hstack(labels)
            preds = np.hstack(preds)
            print("labels:",labels)
            print("preds:",preds)
            read_dic = np.load(fname+"_short_path.npy", allow_pickle=True).item()
            distance_pred = []
            count = 0
            for w in range(len(labels)):
                a = read_dic[labels[w]][preds[w]]
                distance_pred.append(a)
                if labels[w] == preds[w]:
                    count = count + 1
            print('perdict accuracy:', count / len(labels))
            result_dis = {}
            ave_class = 0
            for q in set(distance_pred):
                result_dis[q] = distance_pred.count(q)
                ave_class = q * result_dis[q] + ave_class
            print('CNN distance:', ave_class / len(labels))
    torch.save(net, "models/"+bmname+"_dw.pth")
if __name__ == "__main__":
    main()


