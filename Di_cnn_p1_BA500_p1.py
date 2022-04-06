#cnn数据集生成
#用自己的电脑设给生成数据集大节点数据容易死机
#Di，cnn有向图的;从I开始感染，双向权重
#adj记录成list
#或记录成其他格式/保存为图片
#记录所有jc
#复制的程序，与pro1同时运行节省时间
#感染图改成根据感染节点连接所有边
#同时改变node_labels, 
# adj为NxN;

#propagation5_lei_jian_pro进化，SI模型
#考虑边的权重且按照总数来生成传播图
#propagation_pro1_4.py是单张病毒图的程序

#500节点时，如果同时计算其他方法的每张图的中心，很慢

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

N = 500#记得必须改
fname = "BA"+str(N)
# B = np.load(fname+".npy")#读取固定的图
# G = nx.Graph(B)
G = nx.read_edgelist('./'+fname+'_weight_2.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())

#partition = community_louvain.best_partition(G)
#np.save("BA500_partition.npy",partition)
partition=np.load(fname+'_k5p1'+'.npy',allow_pickle = True).item()
#print(partition[996])
graph_labels = []
graph_labels_class=[]
countedge = []
countadj = []
part=0.5
InfectionRate = 0.5#概率太大，10轮感染1400个节点
Roundtime = 3
adjall = []
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
    
    new_G_small = nx.DiGraph()
    count = [1]
    statechange = []
    edgechange = []
    edgeweight = []
    weight_s = 1
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
                            #new_G_small.add_edge(int(nbr),int(key))     #I->S
        statechange=list(set(statechange))
        for i in statechange:
            S.remove(i)
            I.append(i)
        count.append(len(I))
        statechange = []
        turn = turn+1
        if turn==10:
            if(len(I)==1):
                break
    # #for i in range(Roundtime):
    # while len(I)<=part*len(G.nodes()):
    #     for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
    #         if int(nbr) in S:
    #             node_adj = 0               #S节点的感染邻接点数
    #             for key in datadict:
    #                 if int(key) in I:
    #                     node_adj=node_adj+1
    #                     edgechange.append(int(key))
    #                     edgeweight.append(G.get_edge_data(int(nbr),int(key))['weight'])
    #             for weight in edgeweight:
    #                 weight_s = weight_s*(1-weight)
    #             rate = 1-weight_s
    #             if random.random() <= rate:   #被感染后，节点状态变化，感染图的边增加（周围所有感染节点与该点的连边都算上）
    #                 # for a in edgechange:
    #                 #     new_G_small.add_edge(int(nbr),a)
    #                 statechange.append(int(nbr))
    #             edgechange = []
    #             edgeweight=[]
    #             weight_s=1 
    #     for i in statechange:
    #         S.remove(i)
    #         I.append(i)
    #     for nbr, datadict in G.adj.items():
    #         if int(nbr) in I:
    #             for key in datadict:
    #                 if int(key) in I:
    #                     new_G_small.add_edge(int(nbr),int(key))
    #     # if len(I)==1:    #个别情况下，while可能会陷入死循环
    #     #     break
    #     count.append(len(I))
    #     statechange = []
    new_G_small = G.subgraph(I)#有向图变成图像怎么变
    perfix = os.path.join(datadir,bmname)
    #filename_node_labels = perfix + '_xnode_labels.txt'
    # filename_center = perfix + '_jordancenter.txt'
    # filename_center1 = perfix + '_jordancenterall.txt'
    filename_adj = perfix+'_adjdata'
    # filename_unbet = perfix + '_unbet.txt'
    # filename_discen = perfix + '_discen.txt'
    # filename_dynage = perfix + '_dynage.txt'
    #new_G_small表示感染图
    #new_G表示感染图加上未感染的节点，
    #将new_G中的邻接矩阵扩展到G
    # print(len(new_G_small.edges()))
    # print(len(new_G_small.nodes()))
    new_G = nx.DiGraph()
    new_G.add_nodes_from(i for i in range(N))
    new_G.add_nodes_from(new_G_small.nodes())    #不增加单独节点看实验效果如何，max_nodes=100时，max graph size 是否为100
    new_G.add_edges_from(new_G_small.edges())
    # print(len(new_G.edges()))
    # print(len(new_G.nodes()))
    #Jordan_center  = nx.center(new_G)#非全连接图不能计算

    #if not nx.is_empty(new_G):
    adj_matrix = nx.adjacency_matrix(new_G).todense()
        #Jordan_center  = nx.center(new_G_small)
        # #无偏中介中心性
        # bet_cen = nx.betweenness_centrality(new_G_small)#节点的中介中心性
        # deg = new_G_small.degree()#节点的度
        # ub={}
        # for i in bet_cen.keys():
        #     ub[i] = bet_cen[i]/(math.pow(deg[i],0.85))
        # unbiased_betweenness = max(ub, key=lambda x: ub[x]) 
        # #unbiased_betweenness = ub.index(max(ub))#####list中index不是对应的节点编号，因为节点编号不是连续的。还得用字典
        # with open(filename_unbet,'a') as unbetf:
        #     unbetf.write(str(unbiased_betweenness))
        #     unbetf.write('\n')

        # #distance centrality
        # dis_path = dict(nx.shortest_path_length(new_G_small))
        # dis_d={}
        # for k,v in dis_path.items():
        #     dis_s=0
        #     for i in v.values():
        #         dis_s = dis_s+i
        #     dis_d[k] = dis_s
        # distance_centrality = min(dis_d, key=lambda x: dis_d[x]) 
        # with open(filename_discen,'a') as discenf:
        #     discenf.write(str(distance_centrality))
        #     discenf.write('\n')

        # #dynamic ages
        # frozen_graph = nx.freeze(new_G_small)#G被冷冻为frozen_graph，不会改变
        # unfrozen_graph = nx.Graph(frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
        # AS = nx.adjacency_spectrum(frozen_graph)#邻接矩阵特征值
        # m = np.real(AS).round(4).max()
        # all_nodes = new_G_small.nodes
        # #print(all_nodes)
        # da = {}                             ###!!!!!字典才对
        # for i in all_nodes:
        #     unfrozen_graph.remove_node(i)
        #     AS1 = nx.adjacency_spectrum(unfrozen_graph)
        #     m1 = np.real(AS1).round(4).max()
        #     da[i] = float(format(abs(m-m1)/m,'.4f'))   #单独运算看对不对
        #     unfrozen_graph = nx.Graph(frozen_graph)
        # dynage = max(da, key=lambda x: da[x]) 
        # # dynage=1
        # with open(filename_dynage,'a') as dynagef:
        #     dynagef.write(str(dynage))
        #     dynagef.write('\n')

        # nx.draw(new_G_small,with_labels = True)
        # plt.show()
    #np.savetxt('datatest.txt',adj_matrix)
    #print(adj_matrix.shape)
    countadj_now=adj_matrix.shape[1]
    #countadj.append(adj_matrix.shape[1])
        #countedge.append(new_G.number_of_edges())
        # with open(filename_center,'a') as centerf:
        #     centerf.write(str(choice(Jordan_center)))    #Jordan center随机选，按理来说差别可能不大，实际有差别
        #     centerf.write('\n')
        # with open(filename_center1,'a') as centerf1:
        #     #print(type(Jordan_center))#list
        #     for i in Jordan_center:
        #         centerf1.write(str(i))    #Jordan center
        #         centerf1.write(',')
        #     centerf1.write('\n')
        # with open(filename_node_labels,'a') as labelf:#节点ID作为标签
        #     for i in new_G.nodes:
        #         labelf.write(str(i))
        #         labelf.write('\n')
    graph_labels.append(start_node)
        #graph_labels_class.append(start_node//100)#所属的类   0-99的分类方法
    graph_labels_class.append(partition[start_node])####k-means  反而没；分类方法
    # else:
    #     adj_matrix=[]
    #     countadj_now=[]
    A = nx.to_numpy_matrix(new_G) 
    #adjall.append(A)
    if not os.path.exists(filename_adj):
        os.makedirs(filename_adj)
    number = sn*100+m
    cv2.imwrite(filename_adj+'/'+str(number)+'.png',A)
    # im = Image.fromarray(A)
    # im.convert('1').save(filename_adj+'/'+str(sn)+'.jpeg')
    return (adj_matrix,countadj_now)

#可以自己造邻接矩阵，行和列的范围从adj_matrix.shape开始增加
#直接生成data_A.txt  边的邻接矩阵
def data_A(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_A = perfix + '_A.txt'
    filename_node_labels = perfix + '_dre_node_labels.txt'
    sum_ca_now = 0
    graphs=10
    nodehead=0
    nodetail=N
    # class0= [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 24, 25, 26, 28, 31, 43, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 75, 76, 78, 80, 82, 83, 88, 89, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 109, 144, 166, 170, 172, 174, 175, 177, 178, 180, 181, 182, 183, 184, 186, 188, 202, 203, 205, 207, 208, 211, 212, 213, 214, 215, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 244, 245, 246, 249, 258, 260, 261, 262, 263, 265, 268, 271, 273, 274, 275, 276, 286, 290, 291, 296, 298, 310, 326, 327, 328, 329, 330, 335, 345, 346, 347, 348, 363, 364, 368, 372, 374, 381, 382, 384, 385, 386, 387, 390, 393, 395, 396, 398, 402, 403, 411, 413, 415, 423, 424, 425, 426, 427, 428, 429, 430, 432, 435, 440, 450, 451, 455, 461, 468, 469, 470, 471, 474, 478, 479, 481, 482, 485, 487, 495]
    # class1= [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 64, 69, 74, 77, 79, 85, 99, 102, 110, 111, 134, 138, 161, 162, 163, 164, 167, 168, 169, 171, 173, 191, 192, 193, 194, 198, 204, 247, 252, 256, 259, 264, 267, 269, 270, 272, 281, 282, 283, 292, 293, 294, 299, 302, 303, 304, 305, 306, 315, 323, 325, 349, 350, 367, 369, 370, 371, 376, 377, 383, 394, 401, 416, 417, 418, 419, 420, 421, 449, 452, 453, 454, 458, 459, 473, 475, 476, 486, 490, 493]
    # class2= [33, 34, 35, 36, 37, 38, 49, 112, 113, 116, 118, 119, 120, 121, 122, 124, 126, 127, 129, 130, 131, 132, 135, 136, 137, 139, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 189, 216, 217, 218, 254, 255, 277, 278, 288, 300, 301, 308, 309, 311, 312, 313, 314, 317, 318, 319, 336, 337, 338, 339, 343, 344, 373, 378, 379, 380, 388, 389, 397, 399, 400, 404, 405, 406, 407, 412, 437, 462, 463, 464, 465, 480, 483, 488, 489, 492, 494, 496, 497, 498, 499]
    # class3= [8, 10, 12, 27, 29, 30, 32, 39, 40, 41, 42, 44, 45, 81, 84, 86, 87, 90, 91, 106, 108, 114, 115, 117, 123, 125, 128, 133, 140, 141, 142, 143, 165, 176, 179, 185, 187, 190, 195, 196, 197, 199, 200, 201, 206, 209, 210, 219, 220, 221, 222, 238, 241, 242, 243, 248, 250, 251, 253, 257, 266, 279, 280, 284, 285, 287, 289, 295, 297, 307, 316, 320, 321, 322, 324, 331, 332, 333, 334, 340, 341, 342, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 365, 366, 375, 391, 392, 408, 409, 410, 414, 422, 431, 433, 434, 436, 438, 439, 441, 442, 443, 444, 445, 446, 447, 448, 456, 457, 460, 466, 467, 472, 477, 484, 491]
    # class0= [33, 34, 35, 36, 37, 38, 49, 112, 113, 116, 118, 119, 120, 121, 122, 124, 126, 127, 129, 130, 131, 132, 135, 136, 137, 139, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 189, 216, 217, 218, 254, 255, 277, 278, 288, 300, 301, 308, 309, 311, 312, 313, 314, 317, 318, 319, 336, 337, 338, 339, 343, 344, 373, 378, 379, 380, 388, 389, 397, 399, 400, 404, 405, 406, 407, 412, 437, 462, 463, 464, 465, 480, 483, 488, 489, 492, 494, 496, 497, 498, 499]
    # class1= [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 64, 69, 74, 77, 79, 85, 99, 102, 110, 111, 134, 138, 161, 162, 163, 164, 167, 168, 169, 171, 173, 191, 192, 193, 194, 198, 204, 247, 252, 256, 259, 264, 267, 269, 270, 272, 281, 282, 283, 292, 293, 294, 299, 302, 303, 304, 305, 306, 315, 323, 325, 349, 350, 367, 369, 370, 371, 376, 377, 383, 394, 401, 416, 417, 418, 419, 420, 421, 449, 452, 453, 454, 458, 459, 473, 475, 476, 486, 490, 493]
    # class2= [0, 1, 2, 4, 5, 6, 31, 43, 46, 47, 48, 50, 55, 57, 58, 70, 71, 75, 76, 78, 172, 180, 183, 188, 202, 203, 207, 223, 225, 226, 227, 229, 230, 231, 236, 239, 244, 245, 249, 310, 326, 327, 328, 329, 330, 345, 346, 347, 348, 363, 368, 372, 382, 384, 387, 396, 398, 402, 403, 413, 415, 432, 435, 450, 451, 455, 469, 471, 481, 487]
    # class3= [3, 7, 9, 11, 12, 24, 25, 26, 27, 28, 32, 51, 52, 53, 54, 56, 59, 60, 61, 62, 63, 65, 66, 67, 68, 72, 73, 80, 82, 83, 88, 89, 92, 93, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 108, 109, 144, 165, 166, 170, 174, 175, 176, 177, 178, 181, 182, 184, 186, 205, 206, 208, 211, 212, 213, 214, 215, 224, 228, 232, 233, 234, 235, 237, 238, 240, 241, 246, 258, 260, 261, 262, 263, 265, 268, 271, 273, 274, 275, 276, 279, 280, 284, 286, 287, 290, 291, 296, 297, 298, 316, 324, 331, 332, 333, 335, 352, 353, 356, 361, 362, 364, 374, 381, 385, 386, 390, 393, 395, 408, 409, 410, 411, 423, 424, 425, 426, 427, 428, 429, 430, 436, 440, 447, 460, 461, 468, 470, 472, 474, 478, 479, 482, 485, 495]
    # class4= [8, 10, 29, 30, 39, 40, 41, 42, 44, 45, 81, 84, 86, 87, 90, 91, 106, 114, 115, 117, 123, 125, 128, 133, 140, 141, 142, 143, 179, 185, 187, 190, 195, 196, 197, 199, 200, 201, 209, 210, 219, 220, 221, 222, 242, 243, 248, 250, 251, 253, 257, 266, 285, 289, 295, 307, 320, 321, 322, 334, 340, 341, 342, 351, 354, 355, 357, 358, 359, 360, 365, 366, 375, 391, 392, 414, 422, 431, 433, 434, 438, 439, 441, 442, 443, 444, 445, 446, 448, 456, 457, 466, 467, 477, 484, 491]
    # ###97 , 91 , 70 , 146 , 96
    class0= [0, 2, 4, 5, 6, 8, 18, 24, 25, 31, 33, 46, 60, 63, 67, 69, 71, 78, 87, 88, 90, 91, 100, 113, 114, 120, 127, 133, 135, 140, 142, 151, 153, 154, 156, 161, 165, 167, 179, 183, 186, 201, 203, 204, 205, 213, 214, 222, 223, 225, 230, 231, 235, 237, 240, 249, 250, 259, 261, 273, 280, 282, 288, 289, 290, 295, 297, 300, 301, 303, 304, 309, 311, 312, 314, 319, 321, 330, 333, 345, 346, 348, 354, 355, 356, 358, 367, 369, 373, 376, 377, 378, 379, 383, 386, 387, 393, 396, 398, 402, 413, 414, 419, 421, 422, 424, 427, 429, 431, 438, 439, 445, 448, 449, 455, 458, 459, 464, 469, 482, 486, 490, 495, 496, 499]
    class1= [10, 28, 30, 44, 50, 52, 55, 56, 59, 61, 66, 77, 81, 95, 97, 110, 111, 116, 117, 122, 123, 152, 160, 170, 171, 177, 178, 182, 185, 189, 199, 217, 218, 224, 226, 229, 238, 242, 243, 247, 252, 256, 264, 265, 271, 272, 276, 277, 279, 287, 298, 299, 308, 315, 324, 325, 343, 344, 352, 357, 361, 362, 381, 382, 385, 389, 395, 401, 412, 462, 465, 483]
    class2= [7, 9, 19, 20, 22, 27, 34, 38, 39, 41, 42, 62, 68, 73, 75, 83, 93, 98, 99, 101, 102, 105, 106, 107, 125, 126, 141, 143, 144, 146, 147, 162, 164, 168, 187, 188, 191, 194, 196, 198, 210, 228, 234, 236, 241, 258, 267, 269, 281, 291, 310, 317, 318, 323, 328, 329, 332, 339, 347, 349, 360, 363, 370, 372, 384, 391, 407, 416, 417, 420, 434, 443, 446, 451, 460, 470, 474, 475, 484, 494, 498]
    class3= [1, 3, 11, 12, 14, 15, 17, 23, 32, 37, 40, 45, 47, 48, 53, 58, 70, 72, 80, 82, 85, 89, 92, 103, 108, 112, 115, 118, 119, 124, 134, 138, 139, 145, 148, 155, 159, 163, 169, 172, 174, 175, 176, 181, 184, 190, 193, 195, 197, 202, 206, 208, 211, 212, 215, 219, 221, 233, 239, 244, 246, 248, 253, 257, 260, 266, 270, 275, 285, 286, 292, 302, 305, 306, 307, 313, 320, 322, 331, 336, 337, 338, 342, 351, 353, 359, 364, 365, 371, 374, 375, 390, 392, 399, 405, 406, 410, 418, 425, 428, 432, 433, 436, 437, 440, 442, 447, 450, 456, 463, 467, 468, 471, 472, 473, 479, 481, 485, 487, 488, 489, 491, 492, 493]
    class4= [13, 16, 21, 26, 29, 35, 36, 43, 49, 51, 54, 57, 64, 65, 74, 76, 79, 84, 86, 94, 96, 104, 109, 121, 128, 129, 130, 131, 132, 136, 137, 149, 150, 157, 158, 166, 173, 180, 192, 200, 207, 209, 216, 220, 227, 232, 245, 251, 254, 255, 262, 263, 268, 274, 278, 283, 284, 293, 294, 296, 316, 326, 327, 334, 335, 340, 341, 350, 366, 368, 380, 388, 394, 397, 400, 403, 404, 408, 409, 411, 415, 423, 426, 430, 435, 441, 444, 452, 453, 454, 457, 461, 466, 476, 477, 478, 480, 497]
    #125 , 72 , 81 , 124 , 98
    #nodelist = class0

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

    bmname = fname+'_p0.5_m10_p1_train'
    #path = os.path.join('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data',bmname)
    #path = os.path.join('/home/iot/zcy/usb/copy/rexying_diffpool/diffpool-master/data',bmname)
    path = os.path.join('/home/iot/zcy/usb/copy/my_CNN/cnn_data_dw_3',bmname)
    #path = os.path.join('data',bmname)#调试时生成的文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    perfix = os.path.join(path,bmname)
    filename_readme = perfix+'readme.txt'
    with open(filename_readme,'w') as f:
        f.write('bmname = '+str(bmname)+"\n")
        f.write('N='+str(N)+"\n")
        f.write('底图='+fname+".npy"+"\n")
        f.write(fname+'底图,感染节点占part比例时停止传播,对于有向加权底图'+"\n")
        f.write('part='+str(part)+"\n")
        #f.write('val_datatest'+"\n")

    data=open(filename_readme,'a')
    data_A(path,bmname)
    graph_label(path,bmname)
    #graph_indicator(path,bmname)
    graph_label_classfication(path,bmname)
    #np.save(perfix+ '_adj',adjall)
    # dis_s=0
    # for i in countedge:
    #     dis_s=dis_s+i
    # print('sum of edges:',file=data)
    # print(str(dis_s)+'\n',file=data)
    # s1=0
    # for i in countadj:
    #     s1=s1+i
    # print('sum of adj:',file=data)
    # print(str(s1)+'\n',file=data)

def graph_indicator(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_indic = perfix + '_graph_indicator.txt'
    with open(filename_graph_indic,'a') as f:                   #a追加，w覆盖
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
    with open(filename_graph_labels_class,'a') as f:
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
