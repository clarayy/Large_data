import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
import torch
from scipy.sparse.coo import coo_matrix
import argparse
import math
from random import choice
import community as community_louvain
from PIL import Image
import cv2
N =200#记得必须改
fname = "BA200"
B = np.load(fname+".npy")#读取固定的图
#G = nx.DiGraph(B)    #权重为1的邻接矩阵
G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())
# bmname = 'BA200_p0.1_m50_train'
# path = os.path.join('/home/iot/zcy/usb/copy/myGCN/cnn_data',bmname)
# if not os.path.exists(path):
#     os.makedirs(path)
# perfix = os.path.join(path)
print(len(G.edges()))
A = nx.to_numpy_matrix(G) 
print(A)
#np.save(perfix+'/'+bmname+'.npy',A)