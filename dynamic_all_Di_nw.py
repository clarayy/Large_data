
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
#变量：B，sn范围,每个sn张数 ，Infectionrate, Roundtime，data文件名

#print(nx.__version__)
N =500#记得必须改
fname = "BA"+str(N)
# B = np.load(fname+".npy")#读取固定的图
# G = nx.Graph(B)
G = nx.read_edgelist('./'+fname+'_weight_2.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())

#partition = community_louvain.best_partition(G)
#np.save("BA500_partition.npy",partition)
#partition=np.load(fname+'_p'+'.npy',allow_pickle = True).item()
graph_labels = []
graph_labels_class=[]
countedge = []
countadj = []
part=0.5
InfectionRate = 0.5#概率太大，10轮感染1400个节点
Roundtime = 3
adjall = []
def dynamic_age(new_G_small):
    frozen_graph = nx.freeze(new_G_small)#G被冷冻为frozen_graph，不会改变
    unfrozen_graph = nx.Graph(frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
    AS = nx.adjacency_spectrum(frozen_graph)#邻接矩阵特征值
    before_m = np.real(AS).round(4).max()
    all_nodes = new_G_small.nodes
    #print(all_nodes)
    da = {}                             ###!!!!!字典才对
    for n_i in all_nodes:
        unfrozen_graph.remove_node(n_i)
        AS1 = nx.adjacency_spectrum(unfrozen_graph)
        after_m = np.real(AS1).round(4).max()
        da[n_i] = float(format(abs(before_m-after_m)/before_m,'.4f'))   #单独运算看对不对
        unfrozen_graph = nx.Graph(frozen_graph)
    dynage = max(da, key=lambda x: da[x]) 
    return dynage
def genGraph(sn,datadir,bmname,m):

#感染过程
    node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
    S = node
    I = []

    j=0
    while j<1:
        #start_node = random.choice(node)#1个初始感染节点
        start_node = sn#1个初始感染节点
        I.append(start_node)
        S.remove(start_node)
        j=j+1
    
    print(start_node)
    
    sub_A = nx.DiGraph()#感染过程中逐渐添加边形成的图
    sub_B = nx.DiGraph()#所有I节点连接形成的图
    #new_G_b = nx.DiGraph()#best_weight
    new_G_m = nx.DiGraph()#yuan_weight
    #new_G_w = nx.DiGraph()#worst_weight
    count = [1]
    statechange = []
    edgechange = []
    edgeweight = []
    turn=0
    #for r in range(Roundtime):       #####从I开始遍历，小于边的概率就感染
    while len(I)<=part*len(G.nodes()):
        for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
            if int(nbr) in I:            
                for key in datadict:
                    if int(key) in S:
                        rate = G.get_edge_data(int(nbr),int(key))['weight']   #I->S 
                        if random.random() <= rate: 
                            statechange.append(int(key))
                            sub_A.add_edge(int(nbr),int(key))     #I->S
        statechange=list(set(statechange))
        for i in statechange:
            S.remove(i)
            I.append(i)
        # if len(I)==1:
        #     break
        count.append(len(I))
        statechange = []
        turn = turn+1
        if turn==10:
            if(len(I)==1):
                break
    sub_B = G.subgraph(I)

    for u,v,w in sub_B.edges(data=True):
        #print(w)
        #new_G_b.add_edge(u,v,weight = round(w['weight'],2))
        new_G_m.add_edge(u,v,weight = 1)
        #new_G_w.add_edge(u,v,weight = round(1-w['weight'],2))
    perfix = os.path.join(datadir,bmname)
    #filename_node_labels = perfix + '_xnode_labels.txt'
    filename_adj = perfix+'_adjdata'
    # filename_unbet = perfix + '_unbet.txt'
    # filename_discen = perfix + '_discen.txt'
    #filename_dynage_b = perfix + '_dynage_b.txt'
    filename_dynage_m = perfix + '_dynage_m.txt'
    #filename_dynage_w = perfix + '_dynage_w.txt'
    #new_G_small表示感染图
    #new_G表示感染图加上未感染的节点，
    #将new_G中的邻接矩阵扩展到G
    new_G = nx.DiGraph()    # 在生成图片时用到
    new_G.add_nodes_from(i for i in range(N))
    new_G.add_nodes_from(sub_B.nodes())    #不增加单独节点看实验效果如何，max_nodes=100时，max graph size 是否为100
    new_G.add_edges_from(sub_B.edges(data=True))  #加权的邻接矩阵
    #Jordan_center  = nx.center(new_G)#非全连接图不能计算

    if not nx.is_empty(new_G):
        adj_matrix = nx.adjacency_matrix(new_G).todense()
        #dynamic ages
        countadj_now=adj_matrix.shape[1]
        countadj.append(adj_matrix.shape[1])
        countedge.append(new_G.number_of_edges())

        # dynage_b=dynamic_age(new_G_b)
        # with open(filename_dynage_b,'a') as dynageb:
        #     dynageb.write(str(dynage_b))
        #     dynageb.write('\n')

        dynage_m=dynamic_age(new_G_m)
        with open(filename_dynage_m,'a') as dynagem:
            dynagem.write(str(dynage_m))    #Jordan center随机选，按理来说差别可能不大，实际有差别
            dynagem.write('\n')
        # dynage_w=dynamic_age(new_G_w)
        # with open(filename_dynage_w,'a') as dynagew:
        #     dynagew.write(str(dynage_w))    #Jordan center随机选，按理来说差别可能不大，实际有差别
        #     dynagew.write('\n')

        graph_labels.append(start_node)
        #graph_labels_class.append(start_node//100)#所属的类   0-99的分类方法
        #graph_labels_class.append(partition[start_node])####k-means  反而没；分类方法
    else:
        adj_matrix=[]
        countadj_now=[]
    #二值图片保存
    # A = nx.to_numpy_matrix(new_G) 
    # adjall.append(A)
    # if not os.path.exists(filename_adj):
    #     os.makedirs(filename_adj)
    # number = sn*100+m
    # cv2.imwrite(filename_adj+'/'+str(number)+'.png',A)

    return (adj_matrix,countadj_now)

#可以自己造邻接矩阵，行和列的范围从adj_matrix.shape开始增加
#直接生成data_A.txt  边的邻接矩阵
def data_A(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_A = perfix + '_A.txt'
    filename_node_labels = perfix + '_dre_node_labels.txt'
    sum_ca_now = 0
    graphs=3
    nodehead=0
    nodetail=N
    # class0= [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 24, 25, 26, 28, 31, 43, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 75, 76, 78, 80, 82, 83, 88, 89, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 109, 144, 166, 170, 172, 174, 175, 177, 178, 180, 181, 182, 183, 184, 186, 188, 202, 203, 205, 207, 208, 211, 212, 213, 214, 215, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 244, 245, 246, 249, 258, 260, 261, 262, 263, 265, 268, 271, 273, 274, 275, 276, 286, 290, 291, 296, 298, 310, 326, 327, 328, 329, 330, 335, 345, 346, 347, 348, 363, 364, 368, 372, 374, 381, 382, 384, 385, 386, 387, 390, 393, 395, 396, 398, 402, 403, 411, 413, 415, 423, 424, 425, 426, 427, 428, 429, 430, 432, 435, 440, 450, 451, 455, 461, 468, 469, 470, 471, 474, 478, 479, 481, 482, 485, 487, 495]
    # class1= [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 64, 69, 74, 77, 79, 85, 99, 102, 110, 111, 134, 138, 161, 162, 163, 164, 167, 168, 169, 171, 173, 191, 192, 193, 194, 198, 204, 247, 252, 256, 259, 264, 267, 269, 270, 272, 281, 282, 283, 292, 293, 294, 299, 302, 303, 304, 305, 306, 315, 323, 325, 349, 350, 367, 369, 370, 371, 376, 377, 383, 394, 401, 416, 417, 418, 419, 420, 421, 449, 452, 453, 454, 458, 459, 473, 475, 476, 486, 490, 493]
    # class2= [33, 34, 35, 36, 37, 38, 49, 112, 113, 116, 118, 119, 120, 121, 122, 124, 126, 127, 129, 130, 131, 132, 135, 136, 137, 139, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 189, 216, 217, 218, 254, 255, 277, 278, 288, 300, 301, 308, 309, 311, 312, 313, 314, 317, 318, 319, 336, 337, 338, 339, 343, 344, 373, 378, 379, 380, 388, 389, 397, 399, 400, 404, 405, 406, 407, 412, 437, 462, 463, 464, 465, 480, 483, 488, 489, 492, 494, 496, 497, 498, 499]
    # class3= [8, 10, 12, 27, 29, 30, 32, 39, 40, 41, 42, 44, 45, 81, 84, 86, 87, 90, 91, 106, 108, 114, 115, 117, 123, 125, 128, 133, 140, 141, 142, 143, 165, 176, 179, 185, 187, 190, 195, 196, 197, 199, 200, 201, 206, 209, 210, 219, 220, 221, 222, 238, 241, 242, 243, 248, 250, 251, 253, 257, 266, 279, 280, 284, 285, 287, 289, 295, 297, 307, 316, 320, 321, 322, 324, 331, 332, 333, 334, 340, 341, 342, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 365, 366, 375, 391, 392, 408, 409, 410, 414, 422, 431, 433, 434, 436, 438, 439, 441, 442, 443, 444, 445, 446, 447, 448, 456, 457, 460, 466, 467, 472, 477, 484, 491]
    class0= [33, 34, 35, 36, 37, 38, 49, 112, 113, 116, 118, 119, 120, 121, 122, 124, 126, 127, 129, 130, 131, 132, 135, 136, 137, 139, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 189, 216, 217, 218, 254, 255, 277, 278, 288, 300, 301, 308, 309, 311, 312, 313, 314, 317, 318, 319, 336, 337, 338, 339, 343, 344, 373, 378, 379, 380, 388, 389, 397, 399, 400, 404, 405, 406, 407, 412, 437, 462, 463, 464, 465, 480, 483, 488, 489, 492, 494, 496, 497, 498, 499]
    class1= [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 64, 69, 74, 77, 79, 85, 99, 102, 110, 111, 134, 138, 161, 162, 163, 164, 167, 168, 169, 171, 173, 191, 192, 193, 194, 198, 204, 247, 252, 256, 259, 264, 267, 269, 270, 272, 281, 282, 283, 292, 293, 294, 299, 302, 303, 304, 305, 306, 315, 323, 325, 349, 350, 367, 369, 370, 371, 376, 377, 383, 394, 401, 416, 417, 418, 419, 420, 421, 449, 452, 453, 454, 458, 459, 473, 475, 476, 486, 490, 493]
    class2= [0, 1, 2, 4, 5, 6, 31, 43, 46, 47, 48, 50, 55, 57, 58, 70, 71, 75, 76, 78, 172, 180, 183, 188, 202, 203, 207, 223, 225, 226, 227, 229, 230, 231, 236, 239, 244, 245, 249, 310, 326, 327, 328, 329, 330, 345, 346, 347, 348, 363, 368, 372, 382, 384, 387, 396, 398, 402, 403, 413, 415, 432, 435, 450, 451, 455, 469, 471, 481, 487]
    class3= [3, 7, 9, 11, 12, 24, 25, 26, 27, 28, 32, 51, 52, 53, 54, 56, 59, 60, 61, 62, 63, 65, 66, 67, 68, 72, 73, 80, 82, 83, 88, 89, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 108, 109, 144, 165, 166, 170, 174, 175, 176, 177, 178, 181, 182, 184, 186, 205, 206, 208, 211, 212, 213, 214, 215, 224, 228, 232, 233, 234, 235, 237, 238, 240, 241, 246, 258, 260, 261, 262, 263, 265, 268, 271, 273, 274, 275, 276, 279, 280, 284, 286, 287, 290, 291, 296, 297, 298, 316, 324, 331, 332, 333, 335, 352, 353, 356, 361, 362, 364, 374, 381, 385, 386, 390, 393, 395, 408, 409, 410, 411, 423, 424, 425, 426, 427, 428, 429, 430, 436, 440, 447, 460, 461, 468, 470, 472, 474, 478, 479, 482, 485, 495]
    class4= [8, 10, 29, 30, 39, 40, 41, 42, 44, 45, 81, 84, 86, 87, 90, 91, 106, 114, 115, 117, 123, 125, 128, 133, 140, 141, 142, 143, 179, 185, 187, 190, 195, 196, 197, 199, 200, 201, 209, 210, 219, 220, 221, 222, 242, 243, 248, 250, 251, 253, 257, 266, 285, 289, 295, 307, 320, 321, 322, 334, 340, 341, 342, 351, 354, 355, 357, 358, 359, 360, 365, 366, 375, 391, 392, 414, 422, 431, 433, 434, 438, 439, 441, 442, 443, 444, 445, 446, 448, 456, 457, 466, 467, 477, 484, 491]
    ###97 , 91 , 70 , 146 , 96
    #nodelist = class1

    #for nodesn in nodelist:
    for nodesn in range(nodehead,nodetail):
        for j in range(graphs):
            adj,ca_now=genGraph(nodesn,datadir,bmname,j)
    # with open(filename_A,'w') as f:
    #     with open(filename_node_labels,'w') as f1: #节点度记录
    #         for nodesn in nodelist:
    #         #for nodesn in range(nodehead,nodetail):
    #             for j in range(graphs):
    #                 adj,ca_now=genGraph(nodesn,datadir,bmname,j)
    #                 if len(adj):                      #图为非空，才进行下一步
    #                     coo_A=coo_matrix(adj)   #邻接矩阵的边的行/列的坐标
    #                     edge_index = [coo_A.row,coo_A.col]
    #                     #node_labels(adj)
    #                     a=np.array(adj)
    #                     a=np.sum(a,axis=1)
    #                     a=a.tolist()
    #                     for i in range(len(a)):
    #                         f1.write(str(a[i]))
    #                         f1.write('\n')
    #                     if len(countadj)==1:
    #                         for i in range(len(edge_index[1])):
    #                             f.write(str(coo_A.row[i])+','+str(coo_A.col[i]))
    #                             f.write('\n')
    #                             #print(str(coo_A.row[i])+','+str(coo_A.col[i]))
    #                     else:
    #                         for i in range(len(edge_index[1])):
    #                             f.write(str(coo_A.row[i]+sum_ca_now)+','+str(coo_A.col[i]+sum_ca_now))
    #                             f.write('\n')
    #                             #print(str(coo_A.row[i]+sum_ca_now)+','+str(coo_A.col[i]+sum_ca_now))
    #                     sum_ca_now=sum_ca_now+ca_now
    #/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data
    filename_readme = perfix + 'readme.txt'
    with open(filename_readme,'a') as f:
        #f.write('InfectionRate='+str(InfectionRate)+"\n")
        #f.write('Roundtime='+str(Roundtime)+"\n")
        f.write('[a,b]='+str(nodehead)+','+str(nodetail)+"\n")
        #f.write('nodelist='+str(nodelist)+'\n')
        f.write('every node graphs='+str(graphs)+"\n")


def main():

    bmname = fname+'_p0.5_m3_da'
    #path = os.path.join('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data',bmname)
    #path = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/data',bmname)
    path = os.path.join('/home/iot/zcy/usb/copy/research/graph_python/research/3.18_DW_CNN/data_0.5',bmname)
    #path = os.path.join('data',bmname)#调试时生成的文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    perfix = os.path.join(path,bmname)
    filename_readme = perfix+'readme.txt'
    with open(filename_readme,'w') as f:
        f.write('bmname = '+str(bmname)+"\n")
        f.write('N='+str(N)+"\n")
        f.write('底图='+fname+".npy"+"\n")
        f.write(fname+'底图，感染节点占part比例时停止传播，z=0.1，包括多种对比方法，分5类'+"\n")
        f.write('part='+str(part)+"\n")
        #f.write('val_datatest'+"\n")

    data=open(filename_readme,'a')
    data_A(path,bmname)
    graph_label(path,bmname)
    #graph_indicator(path,bmname)
    #graph_label_classfication(path,bmname)
    #np.save(perfix+ '_adj',adjall)
    dis_s=0
    for i in countedge:
        dis_s=dis_s+i
    print('sum of edges:',file=data)
    print(str(dis_s)+'\n',file=data)
    s1=0
    for i in countadj:
        s1=s1+i
    print('sum of adj:',file=data)
    print(str(s1)+'\n',file=data)

def graph_indicator(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_indic = perfix + '_graph_indicator.txt'
    with open(filename_graph_indic,'w') as f:
        i=1
        for val in countadj:
            for j in range(int(val)):
                f.write(str(i))
                f.write('\n')
            i=i+1
def graph_label(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_labels = perfix+ '_graph_labels.txt'
    with open(filename_graph_labels,'a') as f:
        for i in graph_labels:
            f.write(str(i))
            f.write('\n')
def graph_label_classfication(datadir,bmname): #分类后的节点标签
    perfix = os.path.join(datadir,bmname)
    filename_graph_labels_class = perfix+ '_graph_labels_class.txt'
    with open(filename_graph_labels_class,'w') as f:
        for i in graph_labels_class:
            f.write(str(i))
            f.write('\n')
    # filename_adj = perfix+'_adj.txt'
    # with open(filename_adj,'a') as fadj:
    #     for i in range(10):
    #         for j in range(N):
    #             fadj.write(str(adjall[i][j]))
    #             fadj.write('\n')

if __name__ == "__main__":
    main()
