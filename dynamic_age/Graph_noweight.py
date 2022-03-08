import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
import argparse

N =500
name = "food500"
B = np.load(name+".npy")#读取固定的图
G = nx.Graph(B)
#G = nx.read_edgelist('./'+name+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.Graph())
A = nx.to_numpy_matrix(G) 
print(A)
# pos=nx.spring_layout(G)
# #print(G.edges(data=True))
# nx.draw(G,pos,node_size=1,with_labels = True)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
# plt.show()
InfectionRate = 0.1#概率太大，10轮感染1400个节点
Roundtime =3


#感染过程
node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
#node1 = list(set(node))#节点元素从小到大排序
#print(len(node))
S = node
I = []

j=0
while j<1:
    #start_node = random.choice(node)#1个初始感染节点
    start_node = 0#1个初始感染节点
    I.append(start_node)
    S.remove(start_node)
    j=j+1
print("start_node:")
print(start_node)
part = 0.1
new_G_b = nx.Graph()
new_G_m = nx.Graph()
new_G_w = nx.Graph()
count = [1]
statechange = []
edgechange = []
edgeweight = []
weight_s = 1
#for i in range(Roundtime):
while len(I)<=part*len(G.nodes()):
    for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
        if int(nbr) in S:
            node_adj = 0               #S节点的感染邻接点数
            for key in datadict:
                if int(key) in I:
                    node_adj=node_adj+1
                    edgechange.append(int(key))
                    edgeweight.append(G.get_edge_data(int(nbr),int(key))['weight'])
            for weight in edgeweight:
                weight_s = weight_s*(1-weight)
            rate = 1-weight_s
            if random.random() <= rate:   #被感染后，节点状态变化，感染图的边增加（周围所有感染节点与该点的连边都算上）
                # for a in edgechange:
                #     new_G.add_edge(int(nbr),a)
                statechange.append(int(nbr))
            edgechange = []
            edgeweight=[]
            weight_s=1      
    for i in statechange:
        S.remove(i)
        I.append(i)
    for nbr, datadict in G.adj.items():
        if int(nbr) in I:
            for key in datadict:
                if int(key) in I:
                    new_G_b.add_edge(int(nbr),int(key),weight=G.edges[int(nbr),int(key)]['weight'])
                    new_G_m.add_edge(int(nbr),int(key),weight=1)
                    new_G_w.add_edge(int(nbr),int(key),weight=1-G.edges[int(nbr),int(key)]['weight'])
    if len(I)==1:
        break
    count.append(len(I))
    statechange = []
#print(G.edges())
plt.figure()
Innum = plt.subplot(111)
props = {'title':'Total number of Infected Users',
          'ylabel':'number','xlabel':'time'}
Innum.set(**props)
x = []
for i in range(0,len(count)):
    x.append(i)
Innum.plot(x,count)
Innum.text(0,0,'N:%.0f,Rate:%.4f' %(N,InfectionRate))
print('len(I):',len(I))
print("I=",I)
plt.show()  #显示感染曲线

#显示源节点
#存储边数据
#nx.write_edgelist(new_G,'data_G1.txt',delimiter=',',data = False)
#print(new_G.edges())
#print(len(new_G.edges()))
print('len(new_G_b.nodes()):')
print(len(new_G_b.nodes()))
# print('edges:')
# print(new_G.edges())
print('len(new_G_m.nodes()):')
print(len(new_G_m.nodes()))
# print('new_G_duizhao edges:')
# print(new_G_duizhao.edges())
#nx.draw(new_G)
pos = nx.spring_layout(new_G_b) #为什么报错！！！！！！！！！！
nx.draw(new_G_b,pos,with_labels = True)#逗号是中文的  #画图
labels = nx.get_edge_attributes(new_G_b,'weight')
nx.draw_networkx_edge_labels(new_G_b, pos, edge_labels = labels)
plt.show()
A = nx.to_numpy_matrix(new_G_b) 
print(A)
print(A.shape)
pos = nx.spring_layout(new_G_m) #为什么报错！！！！！！！！！！
nx.draw(new_G_m,pos,with_labels = True)#逗号是中文的  #画图
labels = nx.get_edge_attributes(new_G_m,'weight')
nx.draw_networkx_edge_labels(new_G_m, pos, edge_labels = labels)
plt.show()
A = nx.to_numpy_matrix(new_G_m) 
print(A)
print(A.shape)
#dynamic ages
def dynamic_ages(G):
    frozen_graph = nx.freeze(G)#G被冷冻为frozen_graph，不会改变
    unfrozen_graph = nx.Graph(frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
    AS = nx.adjacency_spectrum(frozen_graph)#邻接矩阵特征值
    m = np.real(AS).round(4).max()
    all_nodes = G.nodes
    #print(all_nodes)
    da = {}                             ###!!!!!字典才对
    for i in all_nodes:
        unfrozen_graph.remove_node(i)
        AS1 = nx.adjacency_spectrum(unfrozen_graph)
        m1 = np.real(AS1).round(4).max()
        da[i] = float(format(abs(m-m1)/m,'.4f'))   #单独运算看对不对
        unfrozen_graph = nx.Graph(frozen_graph)
    dynage = max(da, key=lambda x: da[x]) 
    #print("dynage:",dynage)
    return dynage
print("new_G_b_dynage:",dynamic_ages(new_G_b))
print("new_G_m_dynage:",dynamic_ages(new_G_m))
print("new_G_w_dynage:",dynamic_ages(new_G_w))
# with open(filename_dynage,'a') as dynagef:
#     dynagef.write(str(dynage))
#     dynagef.write('\n')
""" #在一张图上显示
Gt=new_G_b.copy()
for node in I:
    if node!= start_node:
        Gt.remove_node(node)
#print(len(Gt.nodes()))#源节点
#print(len(G.nodes()))
pos = nx.spring_layout(G)
print("len(G.edges():",len(G.edges()))
nx.draw(G,pos,node_color='b',node_size=1,edge_color = 'b',with_labels=True)
nx.draw_networkx_nodes(new_G_b,pos,node_color='r',node_size=1)
nx.draw_networkx_edges(new_G_b,pos,edge_color = 'r')
nx.draw(Gt,pos,node_color = 'red',node_size = 10)
plt.show() """
